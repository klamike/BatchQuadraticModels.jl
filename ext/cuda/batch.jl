# Adapt CPU `BatchQuadraticModel` → GPU. CUDA's default adapt of a scalar
# `SparseMatrixCSC` goes to a dense `CuArray`, so we intercept at the op
# level: route through `_to_cu_csr` (symmetric expansion for Hessian) and
# rebuild with `spmm_ncols = nbatch` so the batched SpMM buffer is premade.
function Adapt.adapt_structure(::Type{<:CuArray}, bqp::BatchQuadraticModel{T, <:Matrix}) where {T}
    nbatch = bqp.meta.nbatch
    meta = _adapt_batch_meta(CuArray, bqp.meta)
    c_batch  = Adapt.adapt(CuArray, bqp.c_batch)
    c0_batch = Adapt.adapt(CuArray, bqp.c0_batch)
    A = _adapt_op_cuda(bqp.A, nbatch; symmetric = false)
    Q = _adapt_op_cuda(bqp.Q, nbatch; symmetric = true)
    _HX = Adapt.adapt(CuArray, bqp._HX)
    _CX = Adapt.adapt(CuArray, bqp._CX)
    return BatchQuadraticModel(meta, c_batch, c0_batch, A, Q, _HX, _CX)
end

# Shared (scalar `SparseOperator`): rebuild on GPU with `spmm_ncols = nbatch`
# so the batched SpMM buffer is premade; Hessian is pre-expanded to full layout.
function _adapt_op_cuda(op::AbstractSparseOperator, nbatch; symmetric)
    scalar = operator_sparse_matrix(op)
    inner  = symmetric ? _to_cu_csr(_expand_symmetric_matrix(scalar)) : _to_cu_csr(scalar)
    return sparse_operator(inner; spmm_ncols = nbatch)
end

# Varying (`BatchSparseOperator`): generic field-wise adapt already lands on a
# `DeviceBatchSparseOperator` via `batch_spmv.jl`.
_adapt_op_cuda(op::BatchSparseOperator, nbatch; symmetric) =
    Adapt.adapt_structure(CuArray, op)
