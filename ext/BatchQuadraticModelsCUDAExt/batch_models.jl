# `BatchQuadraticModel` adapt: CPU sparse ops are rebuilt as `CuSparseOperator`
# with `spmm_ncols = nbatch` so the batched SpMM buffer is premade.

function Adapt.adapt_structure(::Type{<:CuArray}, bqp::BatchQuadraticModel{T, <:Matrix}) where {T}
  nbatch = bqp.meta.nbatch
  return BatchQuadraticModel(
    _adapt_batch_meta(CuArray, bqp.meta),
    Adapt.adapt(CuArray, bqp.c_batch),
    Adapt.adapt(CuArray, bqp.c0_batch),
    _adapt_op_cuda(bqp.A, nbatch; symmetric = false),
    _adapt_op_cuda(bqp.Q, nbatch; symmetric = true),
    Adapt.adapt(CuArray, bqp._HX),
  )
end

# Shared op (scalar `SparseOperator`): rebuild as `CuSparseOperator` with SpMM buffer.
function _adapt_op_cuda(op::AbstractSparseOperator, nbatch; symmetric)
  scalar = operator_sparse_matrix(op)
  inner  = symmetric ? _to_cu_csr(_expand_symmetric_matrix(scalar)) : _to_cu_csr(scalar)
  return sparse_operator(inner; spmm_ncols = nbatch)
end

# Per-instance op (`BatchSparseOperator`): generic field-wise adapt → DeviceBatchSparseOperator.
_adapt_op_cuda(op::BatchSparseOperator, nbatch; symmetric) = Adapt.adapt_structure(CuArray, op)
