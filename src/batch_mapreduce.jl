batch_mapreduce!(f, op, neutral, out::AbstractMatrix, srcs::AbstractMatrix...) =
  (out .= mapreduce(f, op, srcs...; dims = 1, init = neutral))

batch_maximum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, max, typemin(T), out, src)
batch_minimum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, min, typemax(T), out, src)
batch_sum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, +, zero(T), out, src)
