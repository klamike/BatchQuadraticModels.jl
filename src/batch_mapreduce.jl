# ---------- column-wise reductions ----------

"""
    batch_mapreduce!(f, op, neutral, out, srcs...)

Column-wise `mapreduce` over one or more batch matrices: for each batch column
`j`, apply `f` elementwise to `srcs[...][:, j]`, combine with `op` starting
from `neutral`, and write into `out[1, j]`. `out` is `(1, nbatch)` so the
result plugs into any batch consumer expecting a matrix shape.
"""
batch_mapreduce!(f, op, neutral, out::AbstractMatrix, srcs::AbstractMatrix...) =
  (out .= mapreduce(f, op, srcs...; dims = 1, init = neutral))

"""
    batch_maximum!(out, src)
    batch_minimum!(out, src)
    batch_sum!(out, src)

Per-column reductions on a batch matrix. Shorthand for `batch_mapreduce!`
with `identity` and the respective associative op. GPU extension overrides
`batch_mapreduce!`; these wrappers are backend-agnostic.
"""
batch_maximum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, max, typemin(T), out, src)
batch_minimum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, min, typemax(T), out, src)
batch_sum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, +, zero(T), out, src)

# ---------- gather-by-active-set ----------

"""
    gather_columns!(dst, src, roots)

Copy the batch columns indexed by `roots` (the active-instance index vector)
into the first `length(roots)` columns of `dst`. `dst[:, j] = src[:, roots[j]]`.
"""
function gather_columns!(
  dst::AbstractMatrix,
  src::AbstractMatrix,
  roots::AbstractVector{<:Integer},
)
  @assert size(dst, 1) == size(src, 1)
  @assert size(dst, 2) >= length(roots)
  @inbounds for j in eachindex(roots)
    copyto!(view(dst, :, j), view(src, :, Int(roots[j])))
  end
  return dst
end

"""
    gather_entries!(dst, src, roots)

Vector counterpart of [`gather_columns!`](@ref): `dst[j] = src[roots[j]]` for
each `j in eachindex(roots)`.
"""
function gather_entries!(
  dst::AbstractVector,
  src::AbstractVector,
  roots::AbstractVector{<:Integer},
)
  @assert length(dst) >= length(roots)
  @inbounds for j in eachindex(roots)
    dst[j] = src[Int(roots[j])]
  end
  return dst
end
