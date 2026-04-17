@kernel function _fill_sparse_structure!(rows, cols, Ap, Aj, Ax)
  i = @index(Global, Linear)
  for c in Ap[i]:Ap[i + 1] - 1
    rows[c] = i
    cols[c] = Aj[c]
  end
end

function BatchQuadraticModels._copy_sparse_structure!(A::CUSPARSE.CuSparseMatrixCSR, rows::CuVector, cols::CuVector)
  @assert length(cols) == length(rows)
  if length(cols) > 0
    backend = CUDABackend()
    _fill_sparse_structure!(backend)(
      rows, cols,
      A.rowPtr, A.colVal, A.nzVal; ndrange = size(A, 1),
    )
  end
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
  bf .+= bqp.data.c0
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

function Adapt.adapt_structure(::Type{<:CuArray}, bnlp::ObjRHSBatchQuadraticModel{T}) where {T}
  nbatch = bnlp.meta.nbatch
  nvar = bnlp.meta.nvar

  H_raw = operator_sparse_matrix(bnlp.data.Q)
  A_raw = operator_sparse_matrix(bnlp.data.A)
  H_orig_csr = _to_cu_csr(H_raw)
  H_full_csr = _to_cu_csr(_expand_symmetric_matrix(H_raw))
  A_csr = _to_cu_csr(A_raw)

  H_op = _cu_sparse_operator(H_orig_csr, H_full_csr; spmm_ncols = nbatch, premake_spmv = ('N',), premake_spmm = ('N',))
  A_op = sparse_operator(A_csr; symmetric = false, spmm_ncols = nbatch)

  c_gpu = Adapt.adapt(to, bnlp.data.c)
  v_gpu = Adapt.adapt(to, bnlp.data._v)
  data_gpu = QPData(
    A_op,
    c_gpu,
    H_op;
    lcon = Adapt.adapt(to, bnlp.data.lcon),
    ucon = Adapt.adapt(to, bnlp.data.ucon),
    lvar = Adapt.adapt(to, bnlp.data.lvar),
    uvar = Adapt.adapt(to, bnlp.data.uvar),
    c0 = bnlp.data.c0,
    _v = v_gpu,
    regularize = bnlp.data.regularize,
    selected = bnlp.data.selected,
    σ = bnlp.data.σ,
  )

  c_batch_gpu = Adapt.adapt(to, bnlp.c_batch)
  HX_gpu = similar(c_batch_gpu, T, nvar, nbatch)
  CX_gpu = similar(c_batch_gpu, T, nvar, nbatch)
  fill!(HX_gpu, zero(T))
  fill!(CX_gpu, zero(T))

  return data_gpu, c_batch_gpu, HX_gpu, CX_gpu
end
