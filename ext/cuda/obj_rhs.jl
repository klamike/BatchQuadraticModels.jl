@kernel function _fill_sparse_structure!(rows, cols, Ap, Aj)
  i = @index(Global, Linear)
  for c in Ap[i]:Ap[i + 1] - 1
    rows[c] = i
    cols[c] = Aj[c]
  end
end

function BatchQuadraticModels._copy_sparse_structure!(A::CUSPARSE.CuSparseMatrixCSR, rows::CuVector, cols::CuVector)
  @assert length(cols) == length(rows)
  if length(cols) > 0
    _fill_sparse_structure!(CUDABackend())(rows, cols, A.rowPtr, A.colVal; ndrange = size(A, 1))
  end
  return rows, cols
end

function BatchQuadraticModels._copy_sparse_structure!(A::CUSPARSE.CuSparseMatrixCOO, rows::CuVector, cols::CuVector)
  @assert length(cols) == length(rows) == nnz(A)
  copyto!(rows, A.rowInd)
  copyto!(cols, A.colInd)
  return rows, cols
end

function NLPModels.obj!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
  bx::AbstractMatrix{T}, bf::AbstractVector{T},
) where {T, S, M1 <: CuSparseOperator, M2, MT}
  bs = length(bf)
  bf_mat = reshape(bf, 1, bs)
  mul!(bqp._HX, bqp.data.Q, bx)
  bqp._HX .*= T(0.5)
  bqp._HX .+= bqp.c_batch
  batch_mapreduce!(*, +, zero(T), bf_mat, bqp._HX, bx)
  bf .+= bqp.data.c0[]
  return bf
end

function NLPModels.grad!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
  bx::AbstractMatrix{T}, bg::AbstractMatrix{T},
) where {T, S, M1 <: CuSparseOperator, M2, MT}
  mul!(bg, bqp.data.Q, bx)
  bg .+= bqp.c_batch
  return bg
end

function NLPModels.cons!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
  bx::AbstractMatrix{T}, bc::AbstractMatrix{T},
) where {T, S, M1, M2 <: CuSparseOperator, MT}
  mul!(bc, bqp.data.A, bx)
  return bc
end

function NLPModels.jac_structure!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: CuSparseOperator}
  BatchQuadraticModels._copy_sparse_structure!(operator_sparse_matrix(bqp.data.A), jrows, jcols)
  return jrows, jcols
end

function NLPModels.hess_structure!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
) where {T, S, M1 <: CuSparseOperator, M2}
  BatchQuadraticModels._copy_sparse_structure!(operator_sparse_matrix(bqp.data.Q), hrows, hcols)
  return hrows, hcols
end

function NLPModels.jac_coord!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
) where {T, S, M1, M2 <: CuSparseOperator}
  bjvals .= operator_sparse_matrix(bqp.data.A).nzVal
  return bjvals
end

function NLPModels.hess_coord!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) where {T, S, M1 <: CuSparseOperator, M2}
  H = operator_sparse_matrix(bqp.data.Q)
  nnzh = nnz(H)
  nnzh == 0 && return bhvals
  mul!(bhvals, H.nzVal, bobj_weight')
  return bhvals
end

function NLPModels.hprod!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
  bx::AbstractMatrix{T}, by::AbstractMatrix{T}, bv::AbstractMatrix{T},
  bobj_weight::AbstractVector{T}, bHv::AbstractMatrix{T},
) where {T, S, M1 <: CuSparseOperator, M2, MT}
  mul!(bHv, bqp.data.Q, bv)
  bHv .*= bobj_weight'
  return bHv
end

