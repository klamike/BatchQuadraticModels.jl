# Per-column update/recovery kernels for the standard-form path.
#
# Same code drives both the scalar workspace (Vector iterates / 1-column
# `BoundMap`) and the batched workspace (Matrix iterates / multi-column
# `BoundMap`): on a `Vector`, `axes(_, 2) == 1:1` and `v[i, 1]` is well-defined,
# so the matrix loop degenerates cleanly to the scalar case.
#
# GPU overrides live in `ext/cuda/standard_form/{scalar,batch}.jl` and dispatch
# more specifically (`CuVector` / `CuMatrix`) — keep those in sync when
# changing kernel signatures here.

# Gate host-pulling structural checks on `--check-bounds=no`; prod runs elide them.
@inline _should_verify_structure(::NLPModels.AbstractBatchNLPModel) = Base.JLOptions().check_bounds != 2

# Inflate a scalar `BoundMap` into the batched form (matrix `l`/`u`).
# `kind`/`idx`/`row` are shared across the batch; `l`/`u` are stacked columns
# of the orig batch's bound matrices (same backend as `MT`).
function _inflate_bound_map(::Type{MT}, scalar::BoundMap{T}, l::MT, u::MT) where {T, MT <: AbstractMatrix{T}}
  return BoundMap{T, MT, typeof(scalar.idx1), typeof(scalar.kind)}(
    scalar.kind, scalar.idx1, scalar.idx2, copy(l), copy(u), scalar.row,
  )
end

function _update_x_offset!(x_offset::AbstractVecOrMat{T}, meta::BoundMap{T}) where {T}
  @inbounds for j in axes(x_offset, 2), i in eachindex(meta.kind)
    kind = meta.kind[i]
    x_offset[i, j] = kind == VAR_UB   ? meta.u[i, j] :
                     kind == VAR_FREE ? zero(T)      : meta.l[i, j]
  end
  return x_offset
end

# Primary constraint rows: `rhs = isfinite(l) ? l : u` (lower bound when set,
# else upper bound — matches the scalar `_primary_rhs` helper).
function _write_primary_rhs!(rhs_base::AbstractVecOrMat{T}, rows::AbstractVector{<:Integer},
                              l::AbstractVecOrMat{T}, u::AbstractVecOrMat{T}) where {T}
  @inbounds for j in axes(rhs_base, 2), i in eachindex(rows)
    row = rows[i]
    row > 0 || continue
    li = l[i, j]
    rhs_base[row, j] = isfinite(li) ? li : u[i, j]
  end
  return rhs_base
end

# Upper-equality rows (paired with primary rows that had both l and u finite):
# `rhs = u - l` encodes `z + w = u - l` for the slack-plus-upper-complement.
function _write_diff_rhs!(rhs_base::AbstractVecOrMat{T}, rows::AbstractVector{<:Integer},
                           l::AbstractVecOrMat{T}, u::AbstractVecOrMat{T}) where {T}
  @inbounds for j in axes(rhs_base, 2), i in eachindex(rows)
    row = rows[i]
    row > 0 && (rhs_base[row, j] = u[i, j] - l[i, j])
  end
  return rhs_base
end

function _update_rhs_base!(
  rhs_base::AbstractVecOrMat{T},
  var_start::BoundMap{T}, con_start::BoundMap{T},
  var_upper_row::AbstractVector{<:Integer}, con_upper_row::AbstractVector{<:Integer},
) where {T}
  _write_primary_rhs!(rhs_base, con_start.row,   con_start.l, con_start.u)
  _write_diff_rhs!(   rhs_base, var_upper_row,   var_start.l, var_start.u)
  _write_diff_rhs!(   rhs_base, con_upper_row,   con_start.l, con_start.u)
  return rhs_base
end

function _apply_rhs_shift!(rhs::AbstractVecOrMat{T}, rhs_base::AbstractVecOrMat{T},
                            shift::AbstractVecOrMat{T}, rows::AbstractVector{<:Integer}) where {T}
  copyto!(rhs, rhs_base)
  @inbounds for j in axes(rhs, 2)
    for i in eachindex(rows)
      row = rows[i]
      row > 0 && (rhs[row, j] -= shift[i, j])
    end
  end
  return rhs
end

function _update_var_start!(xstd::AbstractVecOrMat{T}, xsrc::AbstractVecOrMat{T}, meta::BoundMap{T}) where {T}
  fill!(xstd, zero(T))
  @inbounds for j in axes(xstd, 2)
    for i in eachindex(meta.kind)
      kind = meta.kind[i]
      if kind == VAR_LB
        xstd[meta.idx1[i], j] = xsrc[i, j] - meta.l[i, j]
      elseif kind == VAR_LB_UB
        xstd[meta.idx1[i], j] = xsrc[i, j] - meta.l[i, j]
        xstd[meta.idx2[i], j] = meta.u[i, j] - xsrc[i, j]
      elseif kind == VAR_UB
        xstd[meta.idx1[i], j] = meta.u[i, j] - xsrc[i, j]
      elseif kind == VAR_FREE
        xi = xsrc[i, j]
        xstd[meta.idx1[i], j] = max(xi, zero(T))
        xstd[meta.idx2[i], j] = max(-xi, zero(T))
      end
    end
  end
  return xstd
end

function _update_constraint_start!(xstd::AbstractVecOrMat{T}, activity::AbstractVecOrMat{T}, meta::BoundMap{T}) where {T}
  @inbounds for j in axes(xstd, 2)
    for i in eachindex(meta.kind)
      kind = meta.kind[i]
      if kind == CON_LB
        xstd[meta.idx1[i], j] = activity[i, j] - meta.l[i, j]
      elseif kind == CON_RANGE
        xstd[meta.idx1[i], j] = activity[i, j] - meta.l[i, j]
        xstd[meta.idx2[i], j] = meta.u[i, j] - activity[i, j]
      elseif kind == CON_UB
        xstd[meta.idx1[i], j] = meta.u[i, j] - activity[i, j]
      end
    end
  end
  return xstd
end

function _update_dual_start!(ystd::AbstractVecOrMat{T}, ysrc::AbstractVecOrMat{T},
                              rows::AbstractVector{<:Integer}) where {T}
  fill!(ystd, zero(T))
  @inbounds for j in axes(ystd, 2)
    for i in eachindex(rows)
      row = rows[i]
      row > 0 && (ystd[row, j] = ysrc[i, j])
    end
  end
  return ystd
end

# `map.base` (length nstd) is broadcast across columns; `src` may be a shared
# vector (same per column) or a per-column matrix. Two methods by src shape:
# the per-column-matrix variant indexes `src[map.src[k], j]`; the shared-vector
# variant reads `src[map.src[k]]` once and reuses it across all columns.
function _apply_scatter_map!(dest::AbstractVecOrMat{T}, map::ScatterMap{T}, src::AbstractMatrix{T}) where {T}
  dest .= map.base
  @inbounds for j in axes(dest, 2)
    for k in eachindex(map.dest)
      dest[map.dest[k], j] += map.scale[k] * src[map.src[k], j]
    end
  end
  return dest
end

function _apply_scatter_map!(dest::AbstractVecOrMat{T}, map::ScatterMap{T}, src::AbstractVector{T}) where {T}
  dest .= map.base
  @inbounds for k in eachindex(map.dest)
    contribution = map.scale[k] * src[map.src[k]]
    for j in axes(dest, 2)
      dest[map.dest[k], j] += contribution
    end
  end
  return dest
end

# `out[j] = sum_i _at(a, i, j) * b[i, j]`. `a` may be a per-column matrix
# (own value per batch column) or a shared vector (broadcast across columns).
@inline _at(a::AbstractVector, i, _) = a[i]
@inline _at(a::AbstractMatrix, i, j) = a[i, j]

function _coldot!(out::AbstractVector{T}, a::AbstractVecOrMat{T}, b::AbstractMatrix{T}) where {T}
  n = size(b, 1)
  @inbounds for j in eachindex(out)
    s = zero(T)
    @simd for i in 1:n
      s += _at(a, i, j) * b[i, j]
    end
    out[j] = s
  end
  return out
end

function _recover_primal_apply!(x::AbstractVecOrMat{T}, kind, idx1, idx2, z::AbstractVecOrMat{T}) where {T}
  @inbounds for j in axes(x, 2)
    for i in axes(x, 1)
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
  return x
end

function _scatter_multipliers!(zl::AbstractVecOrMat{T}, zu::AbstractVecOrMat{T},
                                var_lower, var_upper, zstd::AbstractVecOrMat{T}) where {T}
  fill!(zl, zero(T))
  fill!(zu, zero(T))
  @inbounds for j in axes(zl, 2)
    for i in axes(zl, 1)
      li, ui = var_lower[i], var_upper[i]
      li > 0 && (zl[i, j] = zstd[li, j])
      ui > 0 && (zu[i, j] = zstd[ui, j])
    end
  end
  return zl, zu
end

function _gather_dual!(mult::AbstractVecOrMat{T}, rows, y::AbstractVecOrMat{T}) where {T}
  fill!(mult, zero(T))
  @inbounds for j in axes(mult, 2)
    for i in axes(mult, 1)
      row = rows[i]
      row > 0 && (mult[i, j] = y[row, j])
    end
  end
  return mult
end

# Copy each non-nothing bound/iterate update into the orig meta.
function _absorb_meta!(meta; lvar = nothing, uvar = nothing, lcon = nothing, ucon = nothing, x0 = nothing, y0 = nothing)
  lvar === nothing || copyto!(meta.lvar, lvar)
  uvar === nothing || copyto!(meta.uvar, uvar)
  lcon === nothing || copyto!(meta.lcon, lcon)
  ucon === nothing || copyto!(meta.ucon, ucon)
  x0   === nothing || copyto!(meta.x0,   x0)
  y0   === nothing || copyto!(meta.y0,   y0)
  return meta
end
