# GPU overrides for the standard-form presolve: launches KernelAbstractions
# kernels for every per-element operation and builds Jacobian/Hessian as
# device-side sparse COO matrices.

const _CuSparse{T} = Union{CuSparseMatrixCOO{T}, CuSparseMatrixCSR{T}, CuSparseMatrixCSC{T}}
BatchQuadraticModels._sparse_nzvals(A::_CuSparse) = A.nzVal

_launch!(k!, n::Integer, args...) = n > 0 && k!(CUDABackend())(args...; ndrange = n)

function BatchQuadraticModels._source_structure(A::CuSparseMatrixCOO)
    return A.rowInd, A.colInd
end

function BatchQuadraticModels._source_structure(A::CuSparseMatrixCSR)
    rows = CuVector{Int}(undef, nnz(A))
    cols = similar(rows)
    BatchQuadraticModels._copy_sparse_structure!(A, rows, cols)
    return rows, cols
end

function BatchQuadraticModels._source_structure(A::CuSparseMatrixCSC)
    return BatchQuadraticModels._source_structure(CuSparseMatrixCSR(A))
end

function BatchQuadraticModels._structure_hash(A::CuSparseMatrixCSR)
    return hash((size(A), nnz(A), pointer(A.rowPtr), pointer(A.colVal)))
end

function BatchQuadraticModels._structure_hash(A::CuSparseMatrixCSC)
    return hash((size(A), nnz(A), pointer(A.colPtr), pointer(A.rowVal)))
end

@kernel function _accumulate_map_kernel!(dest, dest_idx, src_idx, scale, src)
    i = @index(Global, Linear)
    @inbounds if i <= length(dest_idx)
        Atomix.@atomic dest[dest_idx[i]] += scale[i] * src[src_idx[i]]
    end
end

function BatchQuadraticModels._apply_scatter_map!(dest::CuVector{T}, map::BatchQuadraticModels.ScatterMap{T}, src::CuVector{T}) where {T}
    copyto!(dest, map.base)
    _launch!(_accumulate_map_kernel!, length(map.dest), dest, map.dest, map.src, map.scale, src)
    return dest
end

@kernel function _recover_primal_kernel!(x, kind, idx1, idx2, z)
    i = @index(Global, Linear)
    @inbounds if i <= length(x)
        k = kind[i]
        if k == BatchQuadraticModels.VAR_LB || k == BatchQuadraticModels.VAR_LB_UB
            x[i] += z[idx1[i]]
        elseif k == BatchQuadraticModels.VAR_UB
            x[i] -= z[idx1[i]]
        elseif k == BatchQuadraticModels.VAR_FREE
            x[i] += z[idx1[i]] - z[idx2[i]]
        end
    end
end

function BatchQuadraticModels._recover_primal_apply!(x::CuVector, kind::CuVector, idx1::CuVector, idx2::CuVector, z::CuVector)
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

function BatchQuadraticModels._gather_dual!(mult::CuVector, rows::CuVector, y::CuVector)
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

function BatchQuadraticModels._scatter_multipliers!(zl::CuVector, zu::CuVector, var_lower::CuVector, var_upper::CuVector, zstd::CuVector)
    _launch!(_scatter_multipliers_kernel!, length(zl), zl, zu, var_lower, var_upper, zstd)
    return zl, zu
end

@kernel function _update_x_offset_kernel!(x_offset, kind, l, u)
    i = @index(Global, Linear)
    @inbounds if i <= length(kind)
        if kind[i] == BatchQuadraticModels.VAR_LB || kind[i] == BatchQuadraticModels.VAR_LB_UB
            x_offset[i] = l[i]
        elseif kind[i] == BatchQuadraticModels.VAR_UB
            x_offset[i] = u[i]
        elseif kind[i] == BatchQuadraticModels.VAR_FREE
            x_offset[i] = zero(eltype(x_offset))
        else
            x_offset[i] = l[i]
        end
    end
end

function BatchQuadraticModels._update_x_offset!(x_offset::CuVector{T}, meta::BatchQuadraticModels.BoundMap{T}) where {T}
    _launch!(_update_x_offset_kernel!, length(meta.kind), x_offset, meta.kind, meta.l, meta.u)
    return x_offset
end

@kernel function _update_var_start_kernel!(xstd, xsrc, kind, idx1, idx2, l, u)
    i = @index(Global, Linear)
    @inbounds if i <= length(kind)
        if kind[i] == BatchQuadraticModels.VAR_LB
            xstd[idx1[i]] = xsrc[i] - l[i]
        elseif kind[i] == BatchQuadraticModels.VAR_LB_UB
            xstd[idx1[i]] = xsrc[i] - l[i]
            xstd[idx2[i]] = u[i] - xsrc[i]
        elseif kind[i] == BatchQuadraticModels.VAR_UB
            xstd[idx1[i]] = u[i] - xsrc[i]
        elseif kind[i] == BatchQuadraticModels.VAR_FREE
            xi = xsrc[i]
            xstd[idx1[i]] = max(xi, zero(eltype(xstd)))
            xstd[idx2[i]] = max(-xi, zero(eltype(xstd)))
        end
    end
end

function BatchQuadraticModels._update_var_start!(xstd::CuVector{T}, xsrc::CuVector{T}, meta::BatchQuadraticModels.BoundMap{T}) where {T}
    fill!(xstd, zero(T))
    _launch!(_update_var_start_kernel!, length(meta.kind), xstd, xsrc, meta.kind, meta.idx1, meta.idx2, meta.l, meta.u)
    return xstd
end

@kernel function _update_con_start_kernel!(xstd, activity, kind, idx1, idx2, l, u)
    i = @index(Global, Linear)
    @inbounds if i <= length(kind)
        if kind[i] == BatchQuadraticModels.CON_LB
            xstd[idx1[i]] = activity[i] - l[i]
        elseif kind[i] == BatchQuadraticModels.CON_RANGE
            xstd[idx1[i]] = activity[i] - l[i]
            xstd[idx2[i]] = u[i] - activity[i]
        elseif kind[i] == BatchQuadraticModels.CON_UB
            xstd[idx1[i]] = u[i] - activity[i]
        end
    end
end

function BatchQuadraticModels._update_constraint_start!(xstd::CuVector{T}, activity::CuVector{T}, meta::BatchQuadraticModels.BoundMap{T}) where {T}
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

function BatchQuadraticModels._update_dual_start!(ystd::CuVector{T}, ysrc::CuVector{T}, rows::CuVector{Int}) where {T}
    fill!(ystd, zero(T))
    _launch!(_update_dual_start_kernel!, length(rows), ystd, ysrc, rows)
    return ystd
end

@kernel function _rhs_base_kernel!(rhs_base, rows, l, u, use_primary)
    i = @index(Global, Linear)
    @inbounds if i <= length(rows)
        row = rows[i]
        if row > 0
            rhs_base[row] = use_primary ? (isfinite(l[i]) ? l[i] : u[i]) : (u[i] - l[i])
        end
    end
end

function BatchQuadraticModels._update_rhs_base!(
    rhs_base::CuVector{T},
    var_start::BatchQuadraticModels.BoundMap{T}, con_start::BatchQuadraticModels.BoundMap{T},
    var_upper_row::CuVector{Int}, con_upper_row::CuVector{Int},
) where {T}
    _launch!(_rhs_base_kernel!, length(con_start.row), rhs_base, con_start.row, con_start.l, con_start.u, true)
    _launch!(_rhs_base_kernel!, length(var_upper_row), rhs_base, var_upper_row, var_start.l, var_start.u, false)
    _launch!(_rhs_base_kernel!, length(con_upper_row), rhs_base, con_upper_row, con_start.l, con_start.u, false)
    return rhs_base
end

@kernel function _apply_rhs_shift_kernel!(rhs, shift, rows)
    i = @index(Global, Linear)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (rhs[row] -= shift[i])
    end
end

function BatchQuadraticModels._apply_rhs_shift!(rhs::CuVector{T}, rhs_base::CuVector{T}, shift::CuVector{T}, rows::CuVector{Int}) where {T}
    copyto!(rhs, rhs_base)
    _launch!(_apply_rhs_shift_kernel!, length(rows), rhs, shift, rows)
    return rhs
end

@inline function _gpu_width(kind::BatchQuadraticModels.BoundKind)
    return kind == BatchQuadraticModels.VAR_FREE ? 2 : kind == BatchQuadraticModels.BK_NONE ? 0 : 1
end

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
        counts[k] = row == 0 ? 0 : _gpu_width(x_kind[cols[k]])
    end
end

@inline function _var_expand(kind::BatchQuadraticModels.BoundKind, idx1::Int, idx2::Int, ::Type{T}) where {T}
    if kind == BatchQuadraticModels.VAR_LB || kind == BatchQuadraticModels.VAR_LB_UB
        return (idx1, 0), (one(T), zero(T)), 1
    elseif kind == BatchQuadraticModels.VAR_UB
        return (idx1, 0), (-one(T), zero(T)), 1
    elseif kind == BatchQuadraticModels.VAR_FREE
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
        counts[k] = _gpu_width(x_kind[rows[k]]) * _gpu_width(x_kind[cols[k]])
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

function _device_c_map(layout::BatchQuadraticModels.StandardFormLayout{T}) where {T}
    vs = layout.var_start
    nc = 0
    @inbounds for i in eachindex(vs.kind)
        nc += BatchQuadraticModels._standard_var_width(vs.kind[i])
    end
    dest = Vector{Int}(undef, nc)
    src = Vector{Int}(undef, nc)
    scale = Vector{T}(undef, nc)
    k = 1
    @inbounds for i in eachindex(vs.kind)
        BatchQuadraticModels._foreach_standard_var(vs, i) do col, s
            dest[k] = col; src[k] = i; scale[k] = s
            k += 1
        end
    end
    return BatchQuadraticModels.ScatterMap(CUDA.zeros(T, layout.nstd), Adapt.adapt(CuArray, dest), Adapt.adapt(CuArray, src), Adapt.adapt(CuArray, scale))
end

function _build_device_jacobian(layout::BatchQuadraticModels.StandardFormLayout{T}, rows::CuVector{Int}, cols::CuVector{Int}) where {T}
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
    return CUSPARSE.CuSparseMatrixCOO(I, J, V, (layout.nrows, layout.nstd)), BatchQuadraticModels.ScatterMap(base, dest, src, scale)
end

function _build_device_hessian(layout::BatchQuadraticModels.StandardFormLayout{T}, rows::CuVector{Int}, cols::CuVector{Int}) where {T}
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
    return CUSPARSE.CuSparseMatrixCOO(I, J, V, (layout.nstd, layout.nstd)), BatchQuadraticModels.ScatterMap(CUDA.zeros(T, total_nnz), dest, src, scale)
end

function _device_workspace(orig, layout::BatchQuadraticModels.StandardFormLayout{T}, A_map, Q_ref, Q_map) where {T}
    y_template = NLPModels.get_y0(orig)
    x_template = NLPModels.get_x0(orig)
    has_q = Q_ref !== nothing
    return BatchQuadraticModels.StandardFormWorkspace(
        orig.data.A,
        similar(BatchQuadraticModels._sparse_nzvals(orig.data.A)),
        A_map,
        _device_c_map(layout),
        BatchQuadraticModels._structure_signature(orig),
        Adapt.adapt(CuArray, layout.rhs),
        Adapt.adapt(CuArray, layout.x_offset),
        Adapt.adapt(CuArray, layout.var_start),
        Adapt.adapt(CuArray, layout.con_start),
        Adapt.adapt(CuArray, layout.var_lower),
        Adapt.adapt(CuArray, layout.var_upper),
        Adapt.adapt(CuArray, layout.var_upper_row),
        Adapt.adapt(CuArray, layout.con_upper_row),
        similar(y_template),
        similar(y_template),
        Q_ref,
        has_q ? similar(BatchQuadraticModels._sparse_nzvals(Q_ref)) : CUDA.zeros(T, 0),
        has_q ? Q_map : BatchQuadraticModels.ScatterMap(CUDA.zeros(T, 0), CUDA.zeros(Int, 0), CUDA.zeros(Int, 0), CUDA.zeros(T, 0)),
        has_q ? similar(x_template) : CUDA.zeros(T, 0),
        has_q ? similar(x_template) : CUDA.zeros(T, 0),
    )
end

function BatchQuadraticModels._build_standard_linear(lp::LinearModel{T, VT, MA}) where {T, VT <: CuVector{T}, MA <: BatchQuadraticModels.AbstractSparseOperator{T}}
    layout = BatchQuadraticModels._build_standard_layout(lp)
    rows, cols = BatchQuadraticModels._source_structure(lp.data.A)
    Astd, A_map = _build_device_jacobian(layout, rows, cols)
    data = LPData(
        BatchQuadraticModels.sparse_operator(Astd; symmetric = false),
        CUDA.zeros(T, layout.nstd);
        lcon = Adapt.adapt(CuArray, layout.rhs),
        ucon = Adapt.adapt(CuArray, layout.rhs),
        lvar = CUDA.zeros(T, layout.nstd),
        uvar = CUDA.fill(T(Inf), layout.nstd),
        c0 = zero(T),
    )
    std = LinearModel(data; x0 = Adapt.adapt(CuArray, layout.x0), y0 = Adapt.adapt(CuArray, layout.y0), minimize = lp.meta.minimize, name = lp.meta.name)
    ws = _device_workspace(lp, layout, A_map, nothing, nothing)
    BatchQuadraticModels.update_standard_form!(lp, std, ws)
    return std, ws
end

function BatchQuadraticModels._build_standard_quadratic(qp::QuadraticModel{T, VT, MQ, MA}) where {T, VT <: CuVector{T}, MQ <: BatchQuadraticModels.AbstractSparseOperator{T}, MA <: BatchQuadraticModels.AbstractSparseOperator{T}}
    layout = BatchQuadraticModels._build_standard_layout(qp)
    A_rows, A_cols = BatchQuadraticModels._source_structure(qp.data.A)
    Q_rows, Q_cols = BatchQuadraticModels._source_structure(qp.data.Q)
    Astd, A_map = _build_device_jacobian(layout, A_rows, A_cols)
    Qstd, Q_map = _build_device_hessian(layout, Q_rows, Q_cols)
    data = QPData(
        BatchQuadraticModels.sparse_operator(Astd; symmetric = false),
        CUDA.zeros(T, layout.nstd),
        BatchQuadraticModels.sparse_operator(Qstd; symmetric = true);
        lcon = Adapt.adapt(CuArray, layout.rhs),
        ucon = Adapt.adapt(CuArray, layout.rhs),
        lvar = CUDA.zeros(T, layout.nstd),
        uvar = CUDA.fill(T(Inf), layout.nstd),
        c0 = zero(T),
        _v = CUDA.zeros(T, layout.nstd),
    )
    std = QuadraticModel(data; x0 = Adapt.adapt(CuArray, layout.x0), y0 = Adapt.adapt(CuArray, layout.y0), minimize = qp.meta.minimize, name = qp.meta.name)
    ws = _device_workspace(qp, layout, A_map, qp.data.Q, Q_map)
    BatchQuadraticModels.update_standard_form!(qp, std, ws)
    return std, ws
end

# ===== Matrix-aware kernels for the batched standard-form path =====
#
# All kernels below dispatch on `CuMatrix` per-instance scratch. The
# `BatchBoundMap` type already carries `MT` for `l` / `u`, so it adapts
# cleanly via `Adapt`.

@kernel function _batch_x_offset_kernel!(x_offset, kind, l, u)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(kind)
        k = kind[i]
        if k == BatchQuadraticModels.VAR_LB || k == BatchQuadraticModels.VAR_LB_UB
            x_offset[i, j] = l[i, j]
        elseif k == BatchQuadraticModels.VAR_UB
            x_offset[i, j] = u[i, j]
        elseif k == BatchQuadraticModels.VAR_FREE
            x_offset[i, j] = zero(eltype(x_offset))
        else
            x_offset[i, j] = l[i, j]
        end
    end
end

function BatchQuadraticModels._batch_update_x_offset!(
    x_offset::CuMatrix{T},
    meta::BatchQuadraticModels.BatchBoundMap{T},
) where {T}
    n = length(meta.kind); bs = size(x_offset, 2)
    n > 0 && _batch_x_offset_kernel!(CUDABackend())(x_offset, meta.kind, meta.l, meta.u; ndrange = (n, bs))
    return x_offset
end

@kernel function _batch_rhs_base_lcon_kernel!(rhs_base, rows, l, u)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(rows)
        row = rows[i]
        if row > 0
            li = l[i, j]
            rhs_base[row, j] = isfinite(li) ? li : u[i, j]
        end
    end
end

@kernel function _batch_rhs_base_diff_kernel!(rhs_base, rows, l, u)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (rhs_base[row, j] = u[i, j] - l[i, j])
    end
end

function BatchQuadraticModels._batch_update_rhs_base!(
    rhs_base::CuMatrix{T},
    var_start::BatchQuadraticModels.BatchBoundMap{T},
    con_start::BatchQuadraticModels.BatchBoundMap{T},
    var_upper_row::CuVector{Int},
    con_upper_row::CuVector{Int},
) where {T}
    bs = size(rhs_base, 2)
    backend = CUDABackend()
    n_con = length(con_start.row)
    n_con > 0 && _batch_rhs_base_lcon_kernel!(backend)(rhs_base, con_start.row, con_start.l, con_start.u; ndrange = (n_con, bs))
    n_vu = length(var_upper_row)
    n_vu > 0 && _batch_rhs_base_diff_kernel!(backend)(rhs_base, var_upper_row, var_start.l, var_start.u; ndrange = (n_vu, bs))
    n_cu = length(con_upper_row)
    n_cu > 0 && _batch_rhs_base_diff_kernel!(backend)(rhs_base, con_upper_row, con_start.l, con_start.u; ndrange = (n_cu, bs))
    return rhs_base
end

@kernel function _batch_apply_rhs_shift_kernel!(rhs, shift, rows)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (rhs[row, j] -= shift[i, j])
    end
end

function BatchQuadraticModels._batch_apply_rhs_shift!(
    rhs::CuMatrix{T}, rhs_base::CuMatrix{T}, shift::CuMatrix{T}, rows::CuVector{Int},
) where {T}
    copyto!(rhs, rhs_base)
    n = length(rows); bs = size(rhs, 2)
    n > 0 && _batch_apply_rhs_shift_kernel!(CUDABackend())(rhs, shift, rows; ndrange = (n, bs))
    return rhs
end

@kernel function _batch_var_start_kernel!(xstd, xsrc, kind, idx1, idx2, l, u)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(kind)
        k = kind[i]
        if k == BatchQuadraticModels.VAR_LB
            xstd[idx1[i], j] = xsrc[i, j] - l[i, j]
        elseif k == BatchQuadraticModels.VAR_LB_UB
            xstd[idx1[i], j] = xsrc[i, j] - l[i, j]
            xstd[idx2[i], j] = u[i, j] - xsrc[i, j]
        elseif k == BatchQuadraticModels.VAR_UB
            xstd[idx1[i], j] = u[i, j] - xsrc[i, j]
        elseif k == BatchQuadraticModels.VAR_FREE
            xi = xsrc[i, j]
            xstd[idx1[i], j] = max(xi, zero(eltype(xstd)))
            xstd[idx2[i], j] = max(-xi, zero(eltype(xstd)))
        end
    end
end

function BatchQuadraticModels._batch_update_var_start!(
    xstd::CuMatrix{T}, xsrc::CuMatrix{T},
    meta::BatchQuadraticModels.BatchBoundMap{T},
) where {T}
    fill!(xstd, zero(T))
    n = length(meta.kind); bs = size(xstd, 2)
    n > 0 && _batch_var_start_kernel!(CUDABackend())(xstd, xsrc, meta.kind, meta.idx1, meta.idx2, meta.l, meta.u; ndrange = (n, bs))
    return xstd
end

@kernel function _batch_con_start_kernel!(xstd, activity, kind, idx1, idx2, l, u)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(kind)
        k = kind[i]
        if k == BatchQuadraticModels.CON_LB
            xstd[idx1[i], j] = activity[i, j] - l[i, j]
        elseif k == BatchQuadraticModels.CON_RANGE
            xstd[idx1[i], j] = activity[i, j] - l[i, j]
            xstd[idx2[i], j] = u[i, j] - activity[i, j]
        elseif k == BatchQuadraticModels.CON_UB
            xstd[idx1[i], j] = u[i, j] - activity[i, j]
        end
    end
end

function BatchQuadraticModels._batch_update_constraint_start!(
    xstd::CuMatrix{T}, activity::CuMatrix{T},
    meta::BatchQuadraticModels.BatchBoundMap{T},
) where {T}
    n = length(meta.kind); bs = size(xstd, 2)
    n > 0 && _batch_con_start_kernel!(CUDABackend())(xstd, activity, meta.kind, meta.idx1, meta.idx2, meta.l, meta.u; ndrange = (n, bs))
    return xstd
end

@kernel function _batch_dual_start_kernel!(ystd, ysrc, rows)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (ystd[row, j] = ysrc[i, j])
    end
end

function BatchQuadraticModels._batch_update_dual_start!(
    ystd::CuMatrix{T}, ysrc::CuMatrix{T}, rows::CuVector{Int},
) where {T}
    fill!(ystd, zero(T))
    n = length(rows); bs = size(ystd, 2)
    n > 0 && _batch_dual_start_kernel!(CUDABackend())(ystd, ysrc, rows; ndrange = (n, bs))
    return ystd
end

@kernel function _batch_scatter_kernel!(dest, dest_idx, src_idx, scale, src)
    k, j = @index(Global, NTuple)
    @inbounds if k <= length(dest_idx)
        Atomix.@atomic dest[dest_idx[k], j] += scale[k] * src[src_idx[k], j]
    end
end

function BatchQuadraticModels._batch_apply_scatter_map!(
    dest::CuMatrix{T}, map::BatchQuadraticModels.ScatterMap{T}, src::CuMatrix{T},
) where {T}
    # `map.base` is a CuVector of length nstd; broadcast to all columns.
    dest .= map.base
    n = length(map.dest); bs = size(dest, 2)
    n > 0 && _batch_scatter_kernel!(CUDABackend())(dest, map.dest, map.src, map.scale, src; ndrange = (n, bs))
    return dest
end

@kernel function _batch_coldot_kernel!(out, a, b, n)
    j = @index(Global, Linear)
    @inbounds if j <= length(out)
        T = eltype(out)
        s = zero(T)
        for i in 1:n
            s += a[i, j] * b[i, j]
        end
        out[j] = s
    end
end

function BatchQuadraticModels._batch_coldot!(
    out::CuVector{T}, a::CuMatrix{T}, b::CuMatrix{T},
) where {T}
    n = size(a, 1); bs = length(out)
    bs > 0 && _batch_coldot_kernel!(CUDABackend())(out, a, b, n; ndrange = bs)
    return out
end

@kernel function _batch_recover_primal_kernel!(x, kind, idx1, idx2, z)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(kind)
        k = kind[i]
        if k == BatchQuadraticModels.VAR_LB || k == BatchQuadraticModels.VAR_LB_UB
            x[i, j] += z[idx1[i], j]
        elseif k == BatchQuadraticModels.VAR_UB
            x[i, j] -= z[idx1[i], j]
        elseif k == BatchQuadraticModels.VAR_FREE
            x[i, j] += z[idx1[i], j] - z[idx2[i], j]
        end
    end
end

function BatchQuadraticModels._batch_recover_primal_apply!(
    x::CuMatrix, kind::CuVector, idx1::CuVector, idx2::CuVector, z::CuMatrix,
)
    n = length(kind); bs = size(x, 2)
    n > 0 && _batch_recover_primal_kernel!(CUDABackend())(x, kind, idx1, idx2, z; ndrange = (n, bs))
    return x
end

@kernel function _batch_scatter_mult_kernel!(zl, zu, var_lower, var_upper, zstd)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(var_lower)
        T = eltype(zl)
        li = var_lower[i]; ui = var_upper[i]
        zl[i, j] = li > 0 ? zstd[li, j] : zero(T)
        zu[i, j] = ui > 0 ? zstd[ui, j] : zero(T)
    end
end

function BatchQuadraticModels._batch_scatter_multipliers!(
    zl::CuMatrix{T}, zu::CuMatrix{T},
    var_lower::CuVector, var_upper::CuVector, zstd::CuMatrix{T},
) where {T}
    fill!(zl, zero(T)); fill!(zu, zero(T))
    n = length(var_lower); bs = size(zl, 2)
    n > 0 && _batch_scatter_mult_kernel!(CUDABackend())(zl, zu, var_lower, var_upper, zstd; ndrange = (n, bs))
    return zl, zu
end

@kernel function _batch_gather_dual_kernel!(mult, rows, y)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(rows)
        row = rows[i]
        T = eltype(mult)
        mult[i, j] = row > 0 ? y[row, j] : zero(T)
    end
end

function BatchQuadraticModels._batch_gather_dual!(
    mult::CuMatrix{T}, rows::CuVector, y::CuMatrix{T},
) where {T}
    n = length(rows); bs = size(mult, 2)
    fill!(mult, zero(T))
    n > 0 && _batch_gather_dual_kernel!(CUDABackend())(mult, rows, y; ndrange = (n, bs))
    return mult
end

# Override `_batch_mul_sparse_symmetric!` for CuSparseOperator (already handled
# via mul!).

# Adapt a host-built scalar std model to GPU before wrapping it in
# `ObjRHSBatchQuadraticModel` for the batch path.
function BatchQuadraticModels._adapt_to_batch_backend(qp::QuadraticModel{T}, ::Type{<:CuMatrix}, nbatch::Int) where {T}
    qp_gpu = Adapt.adapt(CuArray, qp)
    # Rebuild operators with `spmm_ncols = nbatch` so batched SpMM works.
    A_inner = operator_sparse_matrix(qp_gpu.data.A)
    Q_inner = operator_sparse_matrix(qp_gpu.data.Q)
    A_op = sparse_operator(A_inner; spmm_ncols = nbatch)
    Q_op = sparse_operator(Q_inner; symmetric = true, spmm_ncols = nbatch)
    data = QPData(A_op, qp_gpu.data.c, Q_op;
        lcon = qp_gpu.data.lcon, ucon = qp_gpu.data.ucon,
        lvar = qp_gpu.data.lvar, uvar = qp_gpu.data.uvar,
        c0 = qp_gpu.data.c0[], _v = qp_gpu.data._v)
    return QuadraticModel(data; x0 = qp_gpu.meta.x0, y0 = qp_gpu.meta.y0,
        minimize = qp_gpu.meta.minimize, name = qp_gpu.meta.name)
end
function BatchQuadraticModels._adapt_to_batch_backend(lp::LinearModel{T}, ::Type{<:CuMatrix}, nbatch::Int) where {T}
    lp_gpu = Adapt.adapt(CuArray, lp)
    A_inner = operator_sparse_matrix(lp_gpu.data.A)
    A_op = sparse_operator(A_inner; spmm_ncols = nbatch)
    data = LPData(A_op, lp_gpu.data.c;
        lcon = lp_gpu.data.lcon, ucon = lp_gpu.data.ucon,
        lvar = lp_gpu.data.lvar, uvar = lp_gpu.data.uvar,
        c0 = lp_gpu.data.c0[])
    return LinearModel(data; x0 = lp_gpu.meta.x0, y0 = lp_gpu.meta.y0,
        minimize = lp_gpu.meta.minimize, name = lp_gpu.meta.name)
end

