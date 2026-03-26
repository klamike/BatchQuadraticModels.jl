batch_mapreduce!(f, op, neutral, out::AbstractMatrix, srcs::AbstractMatrix...) =
  (out .= mapreduce(f, op, srcs...; dims = 1, init = neutral))
