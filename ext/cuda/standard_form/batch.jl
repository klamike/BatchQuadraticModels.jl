# Batched standard-form GPU overrides — KernelAbstractions kernels for every
# per-column operation that has a CPU loop in `src/standard_form/kernels.jl`.

@inline _launch2d!(k!, n::Integer, bs::Integer, args...) =
  n > 0 && k!(CUDABackend())(args...; ndrange = (n, bs))

@kernel function _batch_x_offset_kernel!(x_offset, kind, l, u)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(kind)
        k = kind[i]
        x_offset[i, j] = k == VAR_UB   ? u[i, j] :
                         k == VAR_FREE ? zero(eltype(x_offset)) : l[i, j]
    end
end

function _update_x_offset!(
    x_offset::CuMatrix{T}, meta::BoundMap{T, <:AbstractMatrix{T}},
) where {T}
    _launch2d!(_batch_x_offset_kernel!, length(meta.kind), size(x_offset, 2),
               x_offset, meta.kind, meta.l, meta.u)
    return x_offset
end

# Primary constraint rows: `rhs = isfinite(l) ? l : u` (lower bound when set,
# else upper bound — matches the scalar `_primary_rhs` helper).
@kernel function _batch_rhs_base_primary_kernel!(rhs_base, rows, l, u)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(rows)
        row = rows[i]
        if row > 0
            li = l[i, j]
            rhs_base[row, j] = isfinite(li) ? li : u[i, j]
        end
    end
end

# Upper-equality rows (paired with primary rows that had both l and u finite):
# `rhs = u - l` encodes `z + w = u - l` for the slack-plus-upper-complement.
@kernel function _batch_rhs_base_diff_kernel!(rhs_base, rows, l, u)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (rhs_base[row, j] = u[i, j] - l[i, j])
    end
end

function _update_rhs_base!(
    rhs_base::CuMatrix{T},
    var_start::BoundMap{T, <:AbstractMatrix{T}}, con_start::BoundMap{T, <:AbstractMatrix{T}},
    var_upper_row::CuVector{Int}, con_upper_row::CuVector{Int},
) where {T}
    bs = size(rhs_base, 2)
    _launch2d!(_batch_rhs_base_primary_kernel!, length(con_start.row), bs, rhs_base, con_start.row, con_start.l, con_start.u)
    _launch2d!(_batch_rhs_base_diff_kernel!,    length(var_upper_row), bs, rhs_base, var_upper_row, var_start.l, var_start.u)
    _launch2d!(_batch_rhs_base_diff_kernel!,    length(con_upper_row), bs, rhs_base, con_upper_row, con_start.l, con_start.u)
    return rhs_base
end

@kernel function _batch_apply_rhs_shift_kernel!(rhs, shift, rows)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (rhs[row, j] -= shift[i, j])
    end
end

function _apply_rhs_shift!(
    rhs::CuMatrix{T}, rhs_base::CuMatrix{T}, shift::CuMatrix{T}, rows::CuVector{Int},
) where {T}
    copyto!(rhs, rhs_base)
    _launch2d!(_batch_apply_rhs_shift_kernel!, length(rows), size(rhs, 2), rhs, shift, rows)
    return rhs
end

@kernel function _batch_var_start_kernel!(xstd, xsrc, kind, idx1, idx2, l, u)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(kind)
        k = kind[i]
        if k == VAR_LB
            xstd[idx1[i], j] = xsrc[i, j] - l[i, j]
        elseif k == VAR_LB_UB
            xstd[idx1[i], j] = xsrc[i, j] - l[i, j]
            xstd[idx2[i], j] = u[i, j] - xsrc[i, j]
        elseif k == VAR_UB
            xstd[idx1[i], j] = u[i, j] - xsrc[i, j]
        elseif k == VAR_FREE
            xi = xsrc[i, j]
            xstd[idx1[i], j] = max(xi, zero(eltype(xstd)))
            xstd[idx2[i], j] = max(-xi, zero(eltype(xstd)))
        end
    end
end

function _update_var_start!(
    xstd::CuMatrix{T}, xsrc::CuMatrix{T}, meta::BoundMap{T, <:AbstractMatrix{T}},
) where {T}
    fill!(xstd, zero(T))
    _launch2d!(_batch_var_start_kernel!, length(meta.kind), size(xstd, 2),
               xstd, xsrc, meta.kind, meta.idx1, meta.idx2, meta.l, meta.u)
    return xstd
end

@kernel function _batch_con_start_kernel!(xstd, activity, kind, idx1, idx2, l, u)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(kind)
        k = kind[i]
        if k == CON_LB
            xstd[idx1[i], j] = activity[i, j] - l[i, j]
        elseif k == CON_RANGE
            xstd[idx1[i], j] = activity[i, j] - l[i, j]
            xstd[idx2[i], j] = u[i, j] - activity[i, j]
        elseif k == CON_UB
            xstd[idx1[i], j] = u[i, j] - activity[i, j]
        end
    end
end

function _update_constraint_start!(
    xstd::CuMatrix{T}, activity::CuMatrix{T}, meta::BoundMap{T, <:AbstractMatrix{T}},
) where {T}
    _launch2d!(_batch_con_start_kernel!, length(meta.kind), size(xstd, 2),
               xstd, activity, meta.kind, meta.idx1, meta.idx2, meta.l, meta.u)
    return xstd
end

@kernel function _batch_dual_start_kernel!(ystd, ysrc, rows)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(rows)
        row = rows[i]
        row > 0 && (ystd[row, j] = ysrc[i, j])
    end
end

function _update_dual_start!(
    ystd::CuMatrix{T}, ysrc::CuMatrix{T}, rows::CuVector{Int},
) where {T}
    fill!(ystd, zero(T))
    _launch2d!(_batch_dual_start_kernel!, length(rows), size(ystd, 2), ystd, ysrc, rows)
    return ystd
end

@kernel function _batch_scatter_kernel!(dest, dest_idx, src_idx, scale, src)
    k, j = @index(Global, NTuple)
    @inbounds if k <= length(dest_idx)
        Atomix.@atomic dest[dest_idx[k], j] += scale[k] * src[src_idx[k], j]
    end
end

# Shared-source variant: same vector applied identically to every batch column.
@kernel function _batch_scatter_shared_kernel!(dest, dest_idx, src_idx, scale, src)
    k, j = @index(Global, NTuple)
    @inbounds if k <= length(dest_idx)
        Atomix.@atomic dest[dest_idx[k], j] += scale[k] * src[src_idx[k]]
    end
end

function _apply_scatter_map!(dest::CuMatrix{T}, map::ScatterMap{T}, src::CuMatrix{T}) where {T}
    dest .= map.base
    _launch2d!(_batch_scatter_kernel!, length(map.dest), size(dest, 2),
               dest, map.dest, map.src, map.scale, src)
    return dest
end

function _apply_scatter_map!(dest::CuMatrix{T}, map::ScatterMap{T}, src::CuVector{T}) where {T}
    dest .= map.base
    _launch2d!(_batch_scatter_shared_kernel!, length(map.dest), size(dest, 2),
               dest, map.dest, map.src, map.scale, src)
    return dest
end

@kernel function _batch_coldot_kernel!(out, a, b, n)
    j = @index(Global, Linear)
    @inbounds if j <= length(out)
        s = zero(eltype(out))
        for i in 1:n
            s += a[i, j] * b[i, j]
        end
        out[j] = s
    end
end

function _coldot!(out::CuVector{T}, a::CuMatrix{T}, b::CuMatrix{T}) where {T}
    bs = length(out)
    bs > 0 && _batch_coldot_kernel!(CUDABackend())(out, a, b, size(a, 1); ndrange = bs)
    return out
end

# Shared-`a` variant for batched standard-form: `out[j] = dot(a, b[:, j])`.
@kernel function _batch_coldot_shared_kernel!(out, a, b, n)
    j = @index(Global, Linear)
    @inbounds if j <= length(out)
        s = zero(eltype(out))
        for i in 1:n
            s += a[i] * b[i, j]
        end
        out[j] = s
    end
end

function _coldot!(out::CuVector{T}, a::CuVector{T}, b::CuMatrix{T}) where {T}
    bs = length(out)
    bs > 0 && _batch_coldot_shared_kernel!(CUDABackend())(out, a, b, length(a); ndrange = bs)
    return out
end

@kernel function _batch_recover_primal_kernel!(x, kind, idx1, idx2, z)
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(kind)
        k = kind[i]
        if k == VAR_LB || k == VAR_LB_UB
            x[i, j] += z[idx1[i], j]
        elseif k == VAR_UB
            x[i, j] -= z[idx1[i], j]
        elseif k == VAR_FREE
            x[i, j] += z[idx1[i], j] - z[idx2[i], j]
        end
    end
end

function _recover_primal_apply!(
    x::AnyCuArray, kind::AnyCuArray, idx1::AnyCuArray, idx2::AnyCuArray, z::AnyCuArray,
)
    _launch2d!(_batch_recover_primal_kernel!, length(kind), size(x, 2), x, kind, idx1, idx2, z)
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

function _scatter_multipliers!(
    zl::AnyCuArray{T}, zu::AnyCuArray{T},
    var_lower::AnyCuArray, var_upper::AnyCuArray, zstd::AnyCuArray{T},
) where {T}
    fill!(zl, zero(T)); fill!(zu, zero(T))
    _launch2d!(_batch_scatter_mult_kernel!, length(var_lower), size(zl, 2),
               zl, zu, var_lower, var_upper, zstd)
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

function _gather_dual!(mult::AnyCuArray{T}, rows::AnyCuArray, y::AnyCuArray{T}) where {T}
    fill!(mult, zero(T))
    _launch2d!(_batch_gather_dual_kernel!, length(rows), size(mult, 2), mult, rows, y)
    return mult
end

# Rebuild GPU sparse operators with `spmm_ncols = nbatch` so the batched SpMM
# buffer is premade for the batched standard-form path.
@inline _spmm_op(op, nbatch; symmetric::Bool) =
    sparse_operator(operator_sparse_matrix(op); symmetric, spmm_ncols = nbatch)

function _adapt_to_batch_backend(qp::QuadraticModel, ::Type{<:CuMatrix}, nbatch::Int)
    gpu = Adapt.adapt(CuArray, qp)
    data = gpu.data
    rebuilt = QPData(_spmm_op(data.A, nbatch; symmetric = false), data.c,
                     _spmm_op(data.Q, nbatch; symmetric = true);
        lcon = data.lcon, ucon = data.ucon, lvar = data.lvar, uvar = data.uvar,
        c0 = data.c0[], _v = data._v)
    return QuadraticModel(rebuilt;
        x0 = gpu.meta.x0, y0 = gpu.meta.y0, minimize = gpu.meta.minimize, name = gpu.meta.name)
end
