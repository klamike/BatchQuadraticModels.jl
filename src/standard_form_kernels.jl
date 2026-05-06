# Per-column kernels — `axes(_, 2) == 1:1` reduces the matrix loop to the
# scalar case. CUDA extension overrides these for CuVector/CuMatrix.

function _update_x_offset!(x_offset::AbstractVecOrMat{T}, meta::BoundMap{T}) where {T}
  @inbounds for j in axes(x_offset, 2), i in eachindex(meta.kind)
    k = meta.kind[i]
    x_offset[i, j] = k == VAR_UB   ? meta.u[i, j] :
                     k == VAR_FREE ? zero(T)      : meta.l[i, j]
  end
  return x_offset
end

function _update_rhs_base!(rhs_base::AbstractVecOrMat{T},
                            var_start::BoundMap{T}, con_start::BoundMap{T},
                            var_upper_row::AbstractVector{<:Integer},
                            con_upper_row::AbstractVector{<:Integer}) where {T}
  @inbounds for j in axes(rhs_base, 2)
    for i in eachindex(con_start.row)
      row = con_start.row[i]; row > 0 || continue
      li = con_start.l[i, j]
      rhs_base[row, j] = isfinite(li) ? li : con_start.u[i, j]
    end
    for i in eachindex(var_upper_row)
      row = var_upper_row[i]
      row > 0 && (rhs_base[row, j] = var_start.u[i, j] - var_start.l[i, j])
    end
    for i in eachindex(con_upper_row)
      row = con_upper_row[i]
      row > 0 && (rhs_base[row, j] = con_start.u[i, j] - con_start.l[i, j])
    end
  end
  return rhs_base
end

function _apply_rhs_shift!(rhs::AbstractVecOrMat{T}, rhs_base::AbstractVecOrMat{T},
                            shift::AbstractVecOrMat{T}, rows::AbstractVector{<:Integer}) where {T}
  copyto!(rhs, rhs_base)
  @inbounds for j in axes(rhs, 2), i in eachindex(rows)
    row = rows[i]; row > 0 && (rhs[row, j] -= shift[i, j])
  end
  return rhs
end

function _update_var_start!(xstd::AbstractVecOrMat{T}, xsrc::AbstractVecOrMat{T},
                             meta::BoundMap{T}) where {T}
  fill!(xstd, zero(T))
  @inbounds for j in axes(xstd, 2), i in eachindex(meta.kind)
    k = meta.kind[i]
    if k == VAR_LB
      xstd[meta.idx1[i], j] = xsrc[i, j] - meta.l[i, j]
    elseif k == VAR_LB_UB
      xstd[meta.idx1[i], j] = xsrc[i, j] - meta.l[i, j]
      xstd[meta.idx2[i], j] = meta.u[i, j] - xsrc[i, j]
    elseif k == VAR_UB
      xstd[meta.idx1[i], j] = meta.u[i, j] - xsrc[i, j]
    elseif k == VAR_FREE
      xi = xsrc[i, j]
      xstd[meta.idx1[i], j] = max(xi, zero(T))
      xstd[meta.idx2[i], j] = max(-xi, zero(T))
    end
  end
  return xstd
end

function _update_constraint_start!(xstd::AbstractVecOrMat{T}, activity::AbstractVecOrMat{T},
                                    meta::BoundMap{T}) where {T}
  @inbounds for j in axes(xstd, 2), i in eachindex(meta.kind)
    k = meta.kind[i]
    if k == CON_LB
      xstd[meta.idx1[i], j] = activity[i, j] - meta.l[i, j]
    elseif k == CON_RANGE
      xstd[meta.idx1[i], j] = activity[i, j] - meta.l[i, j]
      xstd[meta.idx2[i], j] = meta.u[i, j] - activity[i, j]
    elseif k == CON_UB
      xstd[meta.idx1[i], j] = meta.u[i, j] - activity[i, j]
    end
  end
  return xstd
end

function _update_dual_start!(ystd::AbstractVecOrMat{T}, ysrc::AbstractVecOrMat{T},
                              rows::AbstractVector{<:Integer}) where {T}
  fill!(ystd, zero(T))
  @inbounds for j in axes(ystd, 2), i in eachindex(rows)
    row = rows[i]; row > 0 && (ystd[row, j] = ysrc[i, j])
  end
  return ystd
end

function _apply_scatter_map!(dest::AbstractVecOrMat{T}, map::ScatterMap{T},
                              src::AbstractMatrix{T}) where {T}
  dest .= map.base
  @inbounds for j in axes(dest, 2), k in eachindex(map.dest)
    dest[map.dest[k], j] += map.scale[k] * src[map.src[k], j]
  end
  return dest
end

function _apply_scatter_map!(dest::AbstractVecOrMat{T}, map::ScatterMap{T},
                              src::AbstractVector{T}) where {T}
  dest .= map.base
  @inbounds for k in eachindex(map.dest)
    contrib = map.scale[k] * src[map.src[k]]
    for j in axes(dest, 2)
      dest[map.dest[k], j] += contrib
    end
  end
  return dest
end

function _coldot!(out::AbstractVector{T}, a::AbstractVector{T}, b::AbstractMatrix{T}) where {T}
  @inbounds for j in eachindex(out)
    s = zero(T)
    @simd for i in 1:size(b, 1)
      s += a[i] * b[i, j]
    end
    out[j] = s
  end
  return out
end

function _coldot!(out::AbstractVector{T}, a::AbstractMatrix{T}, b::AbstractMatrix{T}) where {T}
  @inbounds for j in eachindex(out)
    s = zero(T)
    @simd for i in 1:size(b, 1)
      s += a[i, j] * b[i, j]
    end
    out[j] = s
  end
  return out
end

function _recover_primal_apply!(x::AbstractVecOrMat{T}, kind, idx1, idx2,
                                 z::AbstractVecOrMat{T}) where {T}
  @inbounds for j in axes(x, 2), i in axes(x, 1)
    k = kind[i]
    if k == VAR_LB || k == VAR_LB_UB
      x[i, j] += z[idx1[i], j]
    elseif k == VAR_UB
      x[i, j] -= z[idx1[i], j]
    elseif k == VAR_FREE
      x[i, j] += z[idx1[i], j] - z[idx2[i], j]
    end
  end
  return x
end

function _scatter_multipliers!(zl::AbstractVecOrMat{T}, zu::AbstractVecOrMat{T},
                                var_lower, var_upper, zstd::AbstractVecOrMat{T}) where {T}
  fill!(zl, zero(T)); fill!(zu, zero(T))
  @inbounds for j in axes(zl, 2), i in axes(zl, 1)
    li, ui = var_lower[i], var_upper[i]
    li > 0 && (zl[i, j] = zstd[li, j])
    ui > 0 && (zu[i, j] = zstd[ui, j])
  end
  return zl, zu
end

function _gather_dual!(mult::AbstractVecOrMat{T}, rows, y::AbstractVecOrMat{T}) where {T}
  fill!(mult, zero(T))
  @inbounds for j in axes(mult, 2), i in axes(mult, 1)
    row = rows[i]; row > 0 && (mult[i, j] = y[row, j])
  end
  return mult
end
