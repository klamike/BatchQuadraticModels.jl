batch_mapreduce!(f, op, neutral, out::AbstractMatrix, srcs::AbstractMatrix...) =
  (out .= mapreduce(f, op, srcs...; dims = 1, init = neutral))

batch_maximum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, max, typemin(T), out, src)
batch_minimum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, min, typemax(T), out, src)
batch_sum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, +, zero(T), out, src)

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
