# GPU overrides for the per-column std-form kernels in
# `src/standard_form_kernels.jl`. Each kernel runs on a 2D ndrange
# `(nrows, batch_size)`; the scalar (CuVector) path reshapes its arguments to
# `(n, 1)` matrices so a single kernel handles both Vec and Mat dispatch.

_sparse_values(A::_CuSparseMatrix) = A.nzVal
_sparse_values(A::CuSparseOperator) = _sparse_values(operator_sparse_matrix(A))
_sparse_structure(A::CuSparseOperator) = _sparse_structure(operator_sparse_matrix(A))

function _sparse_structure(A::CuSparseMatrixCOO)
  return CuVector{Int}(A.rowInd), CuVector{Int}(A.colInd)
end
function _sparse_structure(A::CuSparseMatrixCSR)
  rows = CuVector{Int}(undef, nnz(A)); cols = similar(rows)
  _copy_sparse_structure!(A, rows, cols)
  return rows, cols
end
_sparse_structure(A::CuSparseMatrixCSC) = _sparse_structure(CuSparseMatrixCSR(A))

function _build_scalar_sparse(::CuVector{T}, rows::CuVector, cols::CuVector,
                              vals::CuVector{T}, m::Int, n::Int) where {T <: BlasFloat}
  rows32 = CuVector{Int32}(rows); cols32 = CuVector{Int32}(cols)
  coo = CuSparseMatrixCOO{T, Int32}(rows32, cols32, vals, (m, n), length(vals))
  return sparse_operator(CuSparseMatrixCSC(coo))
end

const _CuVecOrMat{T} = Union{CuVector{T}, CuMatrix{T}}
@inline _as_2d(v::CuVector) = reshape(v, length(v), 1)
@inline _as_2d(m::CuMatrix) = m
@inline _bs(::CuVector) = 1
@inline _bs(m::CuMatrix) = size(m, 2)

_launch2d!(k!, n::Integer, bs::Integer, args...) =
  n > 0 && k!(CUDABackend())(args...; ndrange = (n, bs))


@kernel function _scatter_kernel!(dest, dest_idx, src_idx, scale, src)
  k, j = @index(Global, NTuple)
  @inbounds if k <= length(dest_idx)
    Atomix.@atomic dest[dest_idx[k], j] += scale[k] * src[src_idx[k], j]
  end
end

@kernel function _scatter_shared_kernel!(dest, dest_idx, src_idx, scale, src)
  k, j = @index(Global, NTuple)
  @inbounds if k <= length(dest_idx)
    Atomix.@atomic dest[dest_idx[k], j] += scale[k] * src[src_idx[k]]
  end
end

function _apply_scatter_map!(dest::_CuVecOrMat{T}, map::ScatterMap{T},
                             src::AbstractMatrix{T}) where {T}
  d = _as_2d(dest); d .= map.base
  _launch2d!(_scatter_kernel!, length(map.dest), _bs(dest),
             d, map.dest, map.src, map.scale, _as_2d(src))
  return dest
end

function _apply_scatter_map!(dest::_CuVecOrMat{T}, map::ScatterMap{T},
                             src::AbstractVector{T}) where {T}
  d = _as_2d(dest); d .= map.base
  _launch2d!(_scatter_shared_kernel!, length(map.dest), _bs(dest),
             d, map.dest, map.src, map.scale, src)
  return dest
end


@kernel function _recover_primal_kernel!(x, kind, idx1, idx2, z)
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

function _recover_primal_apply!(x::_CuVecOrMat, kind::AnyCuArray,
                                 idx1::AnyCuArray, idx2::AnyCuArray, z::_CuVecOrMat)
  _launch2d!(_recover_primal_kernel!, length(kind), _bs(x),
             _as_2d(x), kind, idx1, idx2, _as_2d(z))
  return x
end


@kernel function _scatter_mult_kernel!(zl, zu, var_lower, var_upper, zstd)
  i, j = @index(Global, NTuple)
  @inbounds if i <= length(var_lower)
    li = var_lower[i]; ui = var_upper[i]
    T = eltype(zl)
    zl[i, j] = li > 0 ? zstd[li, j] : zero(T)
    zu[i, j] = ui > 0 ? zstd[ui, j] : zero(T)
  end
end

function _scatter_multipliers!(zl::_CuVecOrMat{T}, zu::_CuVecOrMat{T},
                                var_lower::AnyCuArray, var_upper::AnyCuArray,
                                zstd::_CuVecOrMat{T}) where {T}
  fill!(zl, zero(T)); fill!(zu, zero(T))
  _launch2d!(_scatter_mult_kernel!, length(var_lower), _bs(zl),
             _as_2d(zl), _as_2d(zu), var_lower, var_upper, _as_2d(zstd))
  return zl, zu
end

@kernel function _gather_dual_kernel!(mult, rows, y)
  i, j = @index(Global, NTuple)
  @inbounds if i <= length(rows)
    row = rows[i]
    mult[i, j] = row > 0 ? y[row, j] : zero(eltype(mult))
  end
end

function _gather_dual!(mult::_CuVecOrMat{T}, rows::AnyCuArray, y::_CuVecOrMat{T}) where {T}
  fill!(mult, zero(T))
  _launch2d!(_gather_dual_kernel!, length(rows), _bs(mult),
             _as_2d(mult), rows, _as_2d(y))
  return mult
end


@kernel function _x_offset_kernel!(x_offset, kind, l, u)
  i, j = @index(Global, NTuple)
  @inbounds if i <= length(kind)
    k = kind[i]
    x_offset[i, j] = k == VAR_UB   ? u[i, j] :
                     k == VAR_FREE ? zero(eltype(x_offset)) : l[i, j]
  end
end

function _update_x_offset!(x_offset::_CuVecOrMat{T}, meta::BoundMap{T}) where {T}
  _launch2d!(_x_offset_kernel!, length(meta.kind), _bs(x_offset),
             _as_2d(x_offset), meta.kind, _as_2d(meta.l), _as_2d(meta.u))
  return x_offset
end

@kernel function _var_start_kernel!(xstd, xsrc, kind, idx1, idx2, l, u)
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

function _update_var_start!(xstd::_CuVecOrMat{T}, xsrc::_CuVecOrMat{T},
                            meta::BoundMap{T}) where {T}
  fill!(xstd, zero(T))
  _launch2d!(_var_start_kernel!, length(meta.kind), _bs(xstd),
             _as_2d(xstd), _as_2d(xsrc), meta.kind, meta.idx1, meta.idx2,
             _as_2d(meta.l), _as_2d(meta.u))
  return xstd
end

@kernel function _con_start_kernel!(xstd, activity, kind, idx1, idx2, l, u)
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

function _update_constraint_start!(xstd::_CuVecOrMat{T}, activity::_CuVecOrMat{T},
                                   meta::BoundMap{T}) where {T}
  _launch2d!(_con_start_kernel!, length(meta.kind), _bs(xstd),
             _as_2d(xstd), _as_2d(activity), meta.kind, meta.idx1, meta.idx2,
             _as_2d(meta.l), _as_2d(meta.u))
  return xstd
end

@kernel function _dual_start_kernel!(ystd, ysrc, rows)
  i, j = @index(Global, NTuple)
  @inbounds if i <= length(rows)
    row = rows[i]; row > 0 && (ystd[row, j] = ysrc[i, j])
  end
end

function _update_dual_start!(ystd::_CuVecOrMat{T}, ysrc::_CuVecOrMat{T}, rows::CuVector{Int}) where {T}
  fill!(ystd, zero(T))
  _launch2d!(_dual_start_kernel!, length(rows), _bs(ystd),
             _as_2d(ystd), _as_2d(ysrc), rows)
  return ystd
end


@kernel function _rhs_base_primary_kernel!(rhs_base, rows, l, u)
  i, j = @index(Global, NTuple)
  @inbounds if i <= length(rows)
    row = rows[i]
    if row > 0
      li = l[i, j]
      rhs_base[row, j] = isfinite(li) ? li : u[i, j]
    end
  end
end

@kernel function _rhs_base_diff_kernel!(rhs_base, rows, l, u)
  i, j = @index(Global, NTuple)
  @inbounds if i <= length(rows)
    row = rows[i]; row > 0 && (rhs_base[row, j] = u[i, j] - l[i, j])
  end
end

function _update_rhs_base!(rhs_base::_CuVecOrMat{T},
                            var_start::BoundMap{T}, con_start::BoundMap{T},
                            var_upper_row::CuVector{Int}, con_upper_row::CuVector{Int}) where {T}
  rb = _as_2d(rhs_base); bs = _bs(rhs_base)
  vsl, vsu = _as_2d(var_start.l), _as_2d(var_start.u)
  csl, csu = _as_2d(con_start.l), _as_2d(con_start.u)
  _launch2d!(_rhs_base_primary_kernel!, length(con_start.row), bs, rb, con_start.row, csl, csu)
  _launch2d!(_rhs_base_diff_kernel!,    length(var_upper_row), bs, rb, var_upper_row, vsl, vsu)
  _launch2d!(_rhs_base_diff_kernel!,    length(con_upper_row), bs, rb, con_upper_row, csl, csu)
  return rhs_base
end

@kernel function _apply_rhs_shift_kernel!(rhs, shift, rows)
  i, j = @index(Global, NTuple)
  @inbounds if i <= length(rows)
    row = rows[i]; row > 0 && (rhs[row, j] -= shift[i, j])
  end
end

function _apply_rhs_shift!(rhs::_CuVecOrMat{T}, rhs_base::_CuVecOrMat{T},
                            shift::_CuVecOrMat{T}, rows::CuVector{Int}) where {T}
  copyto!(rhs, rhs_base)
  _launch2d!(_apply_rhs_shift_kernel!, length(rows), _bs(rhs),
             _as_2d(rhs), _as_2d(shift), rows)
  return rhs
end


@kernel function _coldot_kernel!(out, a, b, n)
  j = @index(Global, Linear)
  @inbounds if j <= length(out)
    s = zero(eltype(out))
    for i in 1:n
      s += a[i, j] * b[i, j]
    end
    out[j] = s
  end
end

@kernel function _coldot_shared_kernel!(out, a, b, n)
  j = @index(Global, Linear)
  @inbounds if j <= length(out)
    s = zero(eltype(out))
    for i in 1:n
      s += a[i] * b[i, j]
    end
    out[j] = s
  end
end

function _coldot!(out::CuVector{T}, a::CuMatrix{T}, b::CuMatrix{T}) where {T}
  bs = length(out)
  bs > 0 && _coldot_kernel!(CUDABackend())(out, a, b, size(a, 1); ndrange = bs)
  return out
end

function _coldot!(out::CuVector{T}, a::CuVector{T}, b::CuMatrix{T}) where {T}
  bs = length(out)
  bs > 0 && _coldot_shared_kernel!(CUDABackend())(out, a, b, length(a); ndrange = bs)
  return out
end

# `c0[1] += scale * dot(a, b)` via GEMV: keeps the result on device, no host sync.
function _add_dot!(c0::CuVector{T}, a::CuVector{T}, b::CuVector{T}, scale::T) where {T}
  mul!(c0, transpose(reshape(a, length(a), 1)), b, scale, one(T))
  return c0
end


# ---- Layout-build kernels (device-resident `_build_standard_layout`) ----
# Variable classification: emit `kind` and the std-form `width` (0/1/2 slots
# per orig var) plus an indicator for VAR_LB_UB equality rows.

@kernel function _classify_var_kernel!(kind, width, has_upper, lvar, uvar)
  i = @index(Global, Linear)
  @inbounds if i <= length(kind)
    li, ui = lvar[i], uvar[i]
    if li == ui
      kind[i] = BK_NONE; width[i] = 0; has_upper[i] = 0
    elseif isfinite(li) && isfinite(ui)
      kind[i] = VAR_LB_UB; width[i] = 2; has_upper[i] = 1
    elseif isfinite(li)
      kind[i] = VAR_LB; width[i] = 1; has_upper[i] = 0
    elseif isfinite(ui)
      kind[i] = VAR_UB; width[i] = 1; has_upper[i] = 0
    else
      kind[i] = VAR_FREE; width[i] = 2; has_upper[i] = 0
    end
  end
end

@kernel function _classify_con_kernel!(kind, kept, slack_width, has_range, lcon, ucon)
  i = @index(Global, Linear)
  @inbounds if i <= length(kind)
    li, ui = lcon[i], ucon[i]
    fl = isfinite(li); fu = isfinite(ui)
    if !fl && !fu
      kind[i] = BK_NONE; kept[i] = 0; slack_width[i] = 0; has_range[i] = 0
    elseif li == ui
      kind[i] = CON_EQ; kept[i] = 1; slack_width[i] = 0; has_range[i] = 0
    elseif fl && fu
      kind[i] = CON_RANGE; kept[i] = 1; slack_width[i] = 2; has_range[i] = 1
    elseif fl
      kind[i] = CON_LB; kept[i] = 1; slack_width[i] = 1; has_range[i] = 0
    else
      kind[i] = CON_UB; kept[i] = 1; slack_width[i] = 1; has_range[i] = 0
    end
  end
end

# Per-var fill: assigns idx1/idx2/var_lower/var_upper/x_offset, populates x0_std.
@kernel function _layout_var_kernel!(idx1, idx2, var_lower, var_upper, x_offset, x0_std,
                                      kind, var_offset, lvar, uvar, x0_src)
  i = @index(Global, Linear)
  T = eltype(x0_std)
  @inbounds if i <= length(kind)
    k = kind[i]
    if k == BK_NONE
      idx1[i] = 0; idx2[i] = 0
      var_lower[i] = 0; var_upper[i] = 0
      x_offset[i] = lvar[i]
    else
      off = var_offset[i]
      li = lvar[i]; ui = uvar[i]; xi = x0_src[i]
      if k == VAR_LB
        s = off + 1
        idx1[i] = s; idx2[i] = 0
        var_lower[i] = s; var_upper[i] = 0
        x_offset[i] = li
        x0_std[s] = xi - li
      elseif k == VAR_LB_UB
        s = off + 1; w = off + 2
        idx1[i] = s; idx2[i] = w
        var_lower[i] = s; var_upper[i] = w
        x_offset[i] = li
        x0_std[s] = xi - li
        x0_std[w] = ui - xi
      elseif k == VAR_UB
        s = off + 1
        idx1[i] = s; idx2[i] = 0
        var_lower[i] = 0; var_upper[i] = s
        x_offset[i] = ui
        x0_std[s] = ui - xi
      else  # VAR_FREE
        s = off + 1; w = off + 2
        idx1[i] = s; idx2[i] = w
        var_lower[i] = 0; var_upper[i] = 0
        x_offset[i] = zero(T)
        x0_std[s] = max(xi, zero(T))
        x0_std[w] = max(-xi, zero(T))
      end
    end
  end
end

# Per-con fill: assigns idx1/idx2/constraint_rows, populates rhs/y0_std and the
# slack rows of extra_I/J/V.
@kernel function _layout_con_kernel!(idx1, idx2, constraint_rows,
                                      rhs, y0_std, extra_I, extra_J, extra_V,
                                      kind, kept_prefix, slack_prefix, lcon, ucon, y0_src,
                                      nstd_v::Int)
  i = @index(Global, Linear)
  T = eltype(rhs)
  @inbounds if i <= length(kind)
    k = kind[i]
    if k == BK_NONE
      idx1[i] = 0; idx2[i] = 0; constraint_rows[i] = 0
    else
      row = kept_prefix[i] + 1
      constraint_rows[i] = row
      li = lcon[i]; ui = ucon[i]
      rhs[row] = isfinite(li) ? li : ui
      y0_std[row] = y0_src[i]
      if k == CON_EQ
        idx1[i] = 0; idx2[i] = 0
      else
        slack_off = slack_prefix[i]
        s = nstd_v + slack_off + 1
        idx1[i] = s
        eidx = slack_off + 1
        sign_v = (k == CON_UB) ? one(T) : -one(T)
        extra_I[eidx] = row; extra_J[eidx] = s; extra_V[eidx] = sign_v
        if k == CON_RANGE
          w = nstd_v + slack_off + 2
          idx2[i] = w
          extra_I[eidx + 1] = row; extra_J[eidx + 1] = w; extra_V[eidx + 1] = one(T)
        else
          idx2[i] = 0
        end
      end
    end
  end
end

# Per-eq fill: appends `z + w = u - l` rows to extra_I/J/V/rhs, sets *_upper_row.
@kernel function _layout_eq_kernel!(upper_row, extra_I, extra_J, extra_V, rhs,
                                     has_eq, eq_prefix, idx1, idx2, src_l, src_u,
                                     extra_offset::Int, row_offset::Int)
  i = @index(Global, Linear)
  T = eltype(rhs)
  @inbounds if i <= length(has_eq)
    if has_eq[i] == 1
      eq_idx = eq_prefix[i]
      row = row_offset + eq_idx + 1
      upper_row[i] = row
      rhs[row] = src_u[i] - src_l[i]
      base = extra_offset + 2 * eq_idx
      extra_I[base + 1] = row; extra_J[base + 1] = idx1[i]; extra_V[base + 1] = one(T)
      extra_I[base + 2] = row; extra_J[base + 2] = idx2[i]; extra_V[base + 2] = one(T)
    else
      upper_row[i] = 0
    end
  end
end

# c-map count + fill (mirrors `_foreach_standard_var`):
#   VAR_LB / VAR_LB_UB → 1 entry (idx1, +1); VAR_UB → 1 (idx1, -1);
#   VAR_FREE → 2 (idx1, +1) and (idx2, -1); BK_NONE → 0.
@kernel function _cmap_count_kernel!(count, kind)
  i = @index(Global, Linear)
  @inbounds if i <= length(kind)
    k = kind[i]
    count[i] = (k == BK_NONE) ? 0 : (k == VAR_FREE) ? 2 : 1
  end
end

@kernel function _cmap_fill_kernel!(dest, src, scale, kind, idx1, idx2, prefix)
  i = @index(Global, Linear)
  T = eltype(scale)
  @inbounds if i <= length(kind)
    k = kind[i]
    off = prefix[i]
    if k == VAR_LB || k == VAR_LB_UB
      dest[off + 1] = idx1[i]; src[off + 1] = i; scale[off + 1] = one(T)
    elseif k == VAR_UB
      dest[off + 1] = idx1[i]; src[off + 1] = i; scale[off + 1] = -one(T)
    elseif k == VAR_FREE
      dest[off + 1] = idx1[i]; src[off + 1] = i; scale[off + 1] = one(T)
      dest[off + 2] = idx2[i]; src[off + 2] = i; scale[off + 2] = -one(T)
    end
  end
end
