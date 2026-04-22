_sparse_values(A::_CuSparseMatrix) = A.nzVal

_launch!(k!, n::Integer, args...) = n > 0 && k!(CUDABackend())(args...; ndrange = n)

function _sparse_structure(A::CuSparseMatrixCOO)
    # `_coo_to_csr`/`_build_op` dispatch on Int64 CuVector; cast up from CUDA.jl's Int32 storage.
    return CuVector{Int}(A.rowInd), CuVector{Int}(A.colInd)
end

function _sparse_structure(A::CuSparseMatrixCSR)
    rows = CuVector{Int}(undef, nnz(A))
    cols = similar(rows)
    _copy_sparse_structure!(A, rows, cols)
    return rows, cols
end

function _sparse_structure(A::CuSparseMatrixCSC)
    return _sparse_structure(CuSparseMatrixCSR(A))
end

# Forward `CuSparseOperator` to its underlying GPU sparse matrix.
_sparse_values(A::CuSparseOperator)    = _sparse_values(operator_sparse_matrix(A))
_sparse_structure(A::CuSparseOperator) = _sparse_structure(operator_sparse_matrix(A))

# GPU sparse builder for `_representative_qp(::BatchQuadraticModel)`. CUDA.jl
# requires Int32 COO indices; convert through CSC so the result matches the
# `CuSparseOperator{...,CSC,...}` shape expected by the scalar GPU builder.
function _build_scalar_sparse(::CuVector{T}, rows::CuVector, cols::CuVector, vals::CuVector{T}, m::Int, n::Int) where {T <: BlasFloat}
    rows32 = CuVector{Int32}(rows)
    cols32 = CuVector{Int32}(cols)
    coo = CuSparseMatrixCOO{T, Int32}(rows32, cols32, vals, (m, n), length(vals))
    return sparse_operator(CuSparseMatrixCSC(coo))
end

function _structure_hash(A::CuSparseMatrixCSR)
    return hash((size(A), nnz(A), pointer(A.rowPtr), pointer(A.colVal)))
end

function _structure_hash(A::CuSparseMatrixCSC)
    return hash((size(A), nnz(A), pointer(A.colPtr), pointer(A.rowVal)))
end

@kernel function _accumulate_map_kernel!(dest, dest_idx, src_idx, scale, src)
    i = @index(Global, Linear)
    @inbounds if i <= length(dest_idx)
        Atomix.@atomic dest[dest_idx[i]] += scale[i] * src[src_idx[i]]
    end
end

function _apply_scatter_map!(dest::CuVector{T}, map::ScatterMap{T}, src::CuVector{T}) where {T}
    copyto!(dest, map.base)
    _launch!(_accumulate_map_kernel!, length(map.dest), dest, map.dest, map.src, map.scale, src)
    return dest
end

@kernel function _recover_primal_kernel!(x, kind, idx1, idx2, z)
    i = @index(Global, Linear)
    @inbounds if i <= length(x)
        k = kind[i]
        if k == VAR_LB || k == VAR_LB_UB
            x[i] += z[idx1[i]]
        elseif k == VAR_UB
            x[i] -= z[idx1[i]]
        elseif k == VAR_FREE
            x[i] += z[idx1[i]] - z[idx2[i]]
        end
    end
end

function _recover_primal_apply!(x::CuVector, kind::CuVector, idx1::CuVector, idx2::CuVector, z::CuVector)
    _launch!(_recover_primal_kernel!, length(x), x, kind, idx1, idx2, z)
    return x
end

@kernel function _gather_dual_kernel!(mult, rows, y)
    i = @index(Global, Linear)
    @inbounds if i <= length(mult)
        row = rows[i]
        mult[i] = row > 0 ? y[row] : zero(eltype(mult))
    end
end

function _gather_dual!(mult::CuVector, rows::CuVector, y::CuVector)
    _launch!(_gather_dual_kernel!, length(mult), mult, rows, y)
    return mult
end

@kernel function _scatter_multipliers_kernel!(zl, zu, var_lower, var_upper, zstd)
    i = @index(Global, Linear)
    @inbounds if i <= length(zl)
        li = var_lower[i]; ui = var_upper[i]
        T = eltype(zl)
        zl[i] = li > 0 ? zstd[li] : zero(T)
        zu[i] = ui > 0 ? zstd[ui] : zero(T)
    end
end

function _scatter_multipliers!(zl::CuVector, zu::CuVector, var_lower::CuVector, var_upper::CuVector, zstd::CuVector)
    _launch!(_scatter_multipliers_kernel!, length(zl), zl, zu, var_lower, var_upper, zstd)
    return zl, zu
end

@kernel function _update_x_offset_kernel!(x_offset, kind, l, u)
    i = @index(Global, Linear)
    @inbounds if i <= length(kind)
        k = kind[i]
        x_offset[i] = k == VAR_UB   ? u[i] :
                      k == VAR_FREE ? zero(eltype(x_offset)) : l[i]
    end
end

function _update_x_offset!(x_offset::CuVector{T}, meta::BoundMap{T}) where {T}
    _launch!(_update_x_offset_kernel!, length(meta.kind), x_offset, meta.kind, meta.l, meta.u)
    return x_offset
end

@kernel function _update_var_start_kernel!(xstd, xsrc, kind, idx1, idx2, l, u)
    i = @index(Global, Linear)
    @inbounds if i <= length(kind)
        if kind[i] == VAR_LB
            xstd[idx1[i]] = xsrc[i] - l[i]
        elseif kind[i] == VAR_LB_UB
            xstd[idx1[i]] = xsrc[i] - l[i]
            xstd[idx2[i]] = u[i] - xsrc[i]
        elseif kind[i] == VAR_UB
            xstd[idx1[i]] = u[i] - xsrc[i]
        elseif kind[i] == VAR_FREE
            xi = xsrc[i]
            xstd[idx1[i]] = max(xi, zero(eltype(xstd)))
            xstd[idx2[i]] = max(-xi, zero(eltype(xstd)))
        end
    end
end

function _update_var_start!(xstd::CuVector{T}, xsrc::CuVector{T}, meta::BoundMap{T}) where {T}
    fill!(xstd, zero(T))
    _launch!(_update_var_start_kernel!, length(meta.kind), xstd, xsrc, meta.kind, meta.idx1, meta.idx2, meta.l, meta.u)
    return xstd
end

@kernel function _update_con_start_kernel!(xstd, activity, kind, idx1, idx2, l, u)
    i = @index(Global, Linear)
    @inbounds if i <= length(kind)
        if kind[i] == CON_LB
            xstd[idx1[i]] = activity[i] - l[i]
        elseif kind[i] == CON_RANGE
            xstd[idx1[i]] = activity[i] - l[i]
            xstd[idx2[i]] = u[i] - activity[i]
        elseif kind[i] == CON_UB
            xstd[idx1[i]] = u[i] - activity[i]
        end
    end
end

function _update_constraint_start!(xstd::CuVector{T}, activity::CuVector{T}, meta::BoundMap{T}) where {T}
    _launch!(_update_con_start_kernel!, length(meta.kind), xstd, activity, meta.kind, meta.idx1, meta.idx2, meta.l, meta.u)
    return xstd
end

@kernel function _update_dual_start_kernel!(ystd, ysrc, rows)
    i = @index(Global, Linear)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (ystd[row] = ysrc[i])
    end
end

function _update_dual_start!(ystd::CuVector{T}, ysrc::CuVector{T}, rows::CuVector{Int}) where {T}
    fill!(ystd, zero(T))
    _launch!(_update_dual_start_kernel!, length(rows), ystd, ysrc, rows)
    return ystd
end

# Primary constraint rows: `rhs = isfinite(l) ? l : u` (lower bound when set,
# else upper bound — matches the scalar `_primary_rhs` helper).
@kernel function _rhs_base_primary_kernel!(rhs_base, rows, l, u)
    i = @index(Global, Linear)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (rhs_base[row] = isfinite(l[i]) ? l[i] : u[i])
    end
end

# Upper-equality rows (paired with primary rows that had both l and u finite):
# `rhs = u - l` encodes `z + w = u - l` for the slack-plus-upper-complement.
@kernel function _rhs_base_diff_kernel!(rhs_base, rows, l, u)
    i = @index(Global, Linear)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (rhs_base[row] = u[i] - l[i])
    end
end

function _update_rhs_base!(
    rhs_base::CuVector{T},
    var_start::BoundMap{T}, con_start::BoundMap{T},
    var_upper_row::CuVector{Int}, con_upper_row::CuVector{Int},
) where {T}
    _launch!(_rhs_base_primary_kernel!, length(con_start.row), rhs_base, con_start.row, con_start.l, con_start.u)
    _launch!(_rhs_base_diff_kernel!,    length(var_upper_row), rhs_base, var_upper_row, var_start.l, var_start.u)
    _launch!(_rhs_base_diff_kernel!,    length(con_upper_row), rhs_base, con_upper_row, con_start.l, con_start.u)
    return rhs_base
end

@kernel function _apply_rhs_shift_kernel!(rhs, shift, rows)
    i = @index(Global, Linear)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (rhs[row] -= shift[i])
    end
end

function _apply_rhs_shift!(rhs::CuVector{T}, rhs_base::CuVector{T}, shift::CuVector{T}, rows::CuVector{Int}) where {T}
    copyto!(rhs, rhs_base)
    _launch!(_apply_rhs_shift_kernel!, length(rows), rhs, shift, rows)
    return rhs
end


# ---------- GPU build overrides for scalar `standard_form` ----------
# When the input model lives on GPU, the CPU `_build_standard_data` chain
# would scalar-index device-resident `_sparse_structure(...)` outputs. These
# overrides build the std-form Jacobian/Hessian, scatter maps, and workspace
# entirely on device (no GPU→CPU transfers).

# Single scalar pull (one Int) from the prefix-sum tail.
function _device_total(prefix::CuVector{<:Integer})
    isempty(prefix) && return 0
    CUDA.@allowscalar return Int(prefix[end])
end

function _scan_counts(counts::CuVector{Int})
    prefix = similar(counts)
    isempty(counts) || CUDA.scan!(+, prefix, counts; dims = 1)
    return prefix
end

@kernel function _jacobian_counts_kernel!(counts, rows, cols, constraint_rows, x_kind)
    k = @index(Global, Linear)
    @inbounds if k <= length(rows)
        row = constraint_rows[rows[k]]
        counts[k] = row == 0 ? 0 : _standard_var_width(x_kind[cols[k]])
    end
end

@inline function _var_expand(kind::BoundKind, idx1::Int, idx2::Int, ::Type{T}) where {T}
    if kind == VAR_LB || kind == VAR_LB_UB
        return (idx1, 0), (one(T), zero(T)), 1
    elseif kind == VAR_UB
        return (idx1, 0), (-one(T), zero(T)), 1
    elseif kind == VAR_FREE
        return (idx1, idx2), (one(T), -one(T)), 2
    else
        return (0, 0), (zero(T), zero(T)), 0
    end
end

@kernel function _jacobian_fill_kernel!(I, J, V, dest, src, scale, counts, prefix, rows, cols, constraint_rows, x_kind, x_idx1, x_idx2, matrix_offset)
    k = @index(Global, Linear)
    T = eltype(V)
    @inbounds if k <= length(rows) && counts[k] != 0
        count = counts[k]
        row = constraint_rows[rows[k]]
        col = cols[k]
        first = prefix[k] - count + 1
        idxs, scales, cnt = _var_expand(x_kind[col], x_idx1[col], x_idx2[col], T)
        @inbounds for a in 1:cnt
            pos = matrix_offset + first + (a - 1)
            I[pos] = row
            J[pos] = idxs[a]
            V[pos] = scales[a]
            dest[first + (a - 1)] = pos
            src[first + (a - 1)] = k
            scale[first + (a - 1)] = scales[a]
        end
    end
end

@kernel function _hessian_counts_kernel!(counts, rows, cols, x_kind)
    k = @index(Global, Linear)
    @inbounds if k <= length(rows)
        counts[k] = _standard_var_width(x_kind[rows[k]]) * _standard_var_width(x_kind[cols[k]])
    end
end

@kernel function _hessian_fill_kernel!(I, J, V, dest, src, scale, counts, prefix, rows, cols, x_kind, x_idx1, x_idx2)
    k = @index(Global, Linear)
    T = eltype(V)
    @inbounds if k <= length(rows) && counts[k] != 0
        count = counts[k]
        i = rows[k]; j = cols[k]
        start = prefix[k] - count + 1
        idxs_i, scales_i, ci = _var_expand(x_kind[i], x_idx1[i], x_idx2[i], T)
        idxs_j, scales_j, cj = _var_expand(x_kind[j], x_idx1[j], x_idx2[j], T)
        idx = start
        @inbounds for a in 1:ci, b in 1:cj
            ai, aj = idxs_i[a], idxs_j[b]
            sab = scales_i[a] * scales_j[b]
            I[idx] = max(ai, aj)
            J[idx] = min(ai, aj)
            V[idx] = sab
            dest[idx] = idx
            src[idx] = k
            scale[idx] = sab
            idx += 1
        end
    end
end

# c_map for the GPU path (per-orig-var-width destination + scale + src). Built
# via the shared CPU `_build_c_map` then adapted as a one-shot sub-kB transfer
# (layout-only, no orig-data dependence — not GPU→CPU of model state).
_device_c_map(layout::StandardFormLayout) = Adapt.adapt(CuArray, _build_c_map(layout))

function _build_device_jacobian(layout::StandardFormLayout{T}, rows::CuVector{Int}, cols::CuVector{Int}) where {T}
    backend = CUDABackend()
    counts = CuVector{Int}(undef, length(rows))
    if !isempty(rows)
        constraint_rows = Adapt.adapt(CuArray, layout.con_start.row)
        x_kind = Adapt.adapt(CuArray, layout.var_start.kind)
        _jacobian_counts_kernel!(backend)(counts, rows, cols, constraint_rows, x_kind; ndrange = length(rows))
    end
    prefix = _scan_counts(counts)
    source_nnz = _device_total(prefix)
    extra_nnz = length(layout.extra_I)
    total_nnz = extra_nnz + source_nnz

    I = CuVector{Int}(undef, total_nnz)
    J = similar(I)
    V = CuVector{T}(undef, total_nnz)
    base = CUDA.zeros(T, total_nnz)
    if extra_nnz > 0
        copyto!(view(I, 1:extra_nnz), Adapt.adapt(CuArray, layout.extra_I))
        copyto!(view(J, 1:extra_nnz), Adapt.adapt(CuArray, layout.extra_J))
        copyto!(view(V, 1:extra_nnz), Adapt.adapt(CuArray, layout.extra_V))
        copyto!(view(base, 1:extra_nnz), Adapt.adapt(CuArray, layout.extra_V))
    end

    dest = CuVector{Int}(undef, source_nnz)
    src = similar(dest)
    scale = CuVector{T}(undef, source_nnz)
    if source_nnz > 0
        constraint_rows = Adapt.adapt(CuArray, layout.con_start.row)
        x_kind = Adapt.adapt(CuArray, layout.var_start.kind)
        x_idx1 = Adapt.adapt(CuArray, layout.var_start.idx1)
        x_idx2 = Adapt.adapt(CuArray, layout.var_start.idx2)
        _jacobian_fill_kernel!(backend)(
            I, J, V, dest, src, scale, counts, prefix, rows, cols,
            constraint_rows, x_kind, x_idx1, x_idx2, extra_nnz;
            ndrange = length(rows),
        )
    end
    return CUSPARSE.CuSparseMatrixCOO(I, J, V, (layout.nrows, layout.nstd)), ScatterMap(base, dest, src, scale)
end

function _build_device_hessian(layout::StandardFormLayout{T}, rows::CuVector{Int}, cols::CuVector{Int}) where {T}
    backend = CUDABackend()
    counts = CuVector{Int}(undef, length(rows))
    if !isempty(rows)
        x_kind = Adapt.adapt(CuArray, layout.var_start.kind)
        _hessian_counts_kernel!(backend)(counts, rows, cols, x_kind; ndrange = length(rows))
    end
    prefix = _scan_counts(counts)
    total_nnz = _device_total(prefix)
    I = CuVector{Int}(undef, total_nnz)
    J = similar(I)
    V = CuVector{T}(undef, total_nnz)
    dest = CuVector{Int}(undef, total_nnz)
    src = similar(dest)
    scale = CuVector{T}(undef, total_nnz)
    if total_nnz > 0
        x_kind = Adapt.adapt(CuArray, layout.var_start.kind)
        x_idx1 = Adapt.adapt(CuArray, layout.var_start.idx1)
        x_idx2 = Adapt.adapt(CuArray, layout.var_start.idx2)
        _hessian_fill_kernel!(backend)(
            I, J, V, dest, src, scale, counts, prefix, rows, cols, x_kind, x_idx1, x_idx2;
            ndrange = length(rows),
        )
    end
    return CUSPARSE.CuSparseMatrixCOO(I, J, V, (layout.nstd, layout.nstd)), ScatterMap(CUDA.zeros(T, total_nnz), dest, src, scale)
end

function _device_workspace(orig, layout::StandardFormLayout{T}, A_map, Q_ref, Q_map) where {T}
    y_template = NLPModels.get_y0(orig)
    if Q_ref !== nothing
        x_template = NLPModels.get_x0(orig)
        Q_src = similar(_sparse_values(Q_ref))
        qx = similar(x_template); ctmp = similar(x_template)
    else
        Q_src = CUDA.zeros(T, 0)
        Q_map = ScatterMap(CUDA.zeros(T, 0), CUDA.zeros(Int, 0), CUDA.zeros(Int, 0), CUDA.zeros(T, 0))
        qx = CUDA.zeros(T, 0); ctmp = CUDA.zeros(T, 0)
    end
    return StandardFormWorkspace(
        orig.data.A, similar(_sparse_values(orig.data.A)), A_map,
        _device_c_map(layout), _structure_signature(orig),
        Adapt.adapt(CuArray, layout.rhs), Adapt.adapt(CuArray, layout.x_offset),
        Adapt.adapt(CuArray, layout.var_start), Adapt.adapt(CuArray, layout.con_start),
        Adapt.adapt(CuArray, layout.var_lower), Adapt.adapt(CuArray, layout.var_upper),
        Adapt.adapt(CuArray, layout.var_upper_row), Adapt.adapt(CuArray, layout.con_upper_row),
        similar(y_template), similar(y_template),
        Q_ref, Q_src, Q_map, qx, ctmp,
    )
end

# Direct GPU build entry points: dispatched on `VT <: CuVector` so they take
# precedence over the generic `standard_form(::_ScalarModel)` in scalar.jl.
# LP and QP share the layout/jacobian/wrap/ws path; QP adds a Hessian build.
function standard_form(orig::Union{
    LinearModel{T, <:CuVector{T}, <:AbstractSparseOperator{T}},
    QuadraticModel{T, <:CuVector{T}, <:AbstractSparseOperator{T}, <:AbstractSparseOperator{T}},
}) where {T}
    layout = _build_standard_layout(orig)
    A_rows, A_cols = _sparse_structure(orig.data.A)
    Astd, A_map = _build_device_jacobian(layout, A_rows, A_cols)
    rhs_dev = Adapt.adapt(CuArray, layout.rhs)
    Astd_op = sparse_operator(Astd; symmetric = false)
    nstd    = layout.nstd

    if orig isa QuadraticModel
        Q_rows, Q_cols = _sparse_structure(orig.data.Q)
        Qstd, Q_map = _build_device_hessian(layout, Q_rows, Q_cols)
        data = QPData(Astd_op, CUDA.zeros(T, nstd), sparse_operator(Qstd; symmetric = true);
                      lcon = rhs_dev, ucon = rhs_dev,
                      lvar = CUDA.zeros(T, nstd), uvar = CUDA.fill(T(Inf), nstd),
                      c0 = zero(T), _v = CUDA.zeros(T, nstd))
        std = QuadraticModel(data; x0 = Adapt.adapt(CuArray, layout.x0), y0 = Adapt.adapt(CuArray, layout.y0),
                             minimize = orig.meta.minimize, name = orig.meta.name)
        ws = _device_workspace(orig, layout, A_map, orig.data.Q, Q_map)
    else
        data = LPData(Astd_op, CUDA.zeros(T, nstd);
                      lcon = rhs_dev, ucon = rhs_dev,
                      lvar = CUDA.zeros(T, nstd), uvar = CUDA.fill(T(Inf), nstd), c0 = zero(T))
        std = LinearModel(data; x0 = Adapt.adapt(CuArray, layout.x0), y0 = Adapt.adapt(CuArray, layout.y0),
                          minimize = orig.meta.minimize, name = orig.meta.name)
        ws = _device_workspace(orig, layout, A_map, nothing, nothing)
    end
    update_standard_form!(orig, std, ws)
    return std, ws
end
