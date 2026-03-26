@kernel function _fill_sparse_structure!(rows, cols, Ap, Aj, Ax)
  i = @index(Global, Linear)
  for c in Ap[i]:Ap[i + 1] - 1
    rows[c] = i
    cols[c] = Aj[c]
  end
end

function fill_structure!(A::CUSPARSE.CuSparseMatrixCSR, rows, cols)
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
) where {T, S, M1 <: AbstractSparseOperator, M2, MT}
  bs = length(bf)
  bf_mat = reshape(bf, 1, bs)
  if !bqp.meta.islp
    mul!(bqp._HX, bqp.data.H, bx)
    bqp._HX .*= T(0.5)
    bqp._HX .+= bqp.c_batch
    batch_mapreduce!(*, +, zero(T), bf_mat, bqp._HX, bx)
  else
    batch_mapreduce!(*, +, zero(T), bf_mat, bqp.c_batch, bx)
  end
  bf .+= bqp.data.c0
  return bf
end

function NLPModels.grad!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
  bx::AbstractMatrix{T}, bg::AbstractMatrix{T},
) where {T, S, M1 <: AbstractSparseOperator, M2, MT}
  if !bqp.meta.islp
    mul!(bg, bqp.data.H, bx)
    bg .+= bqp.c_batch
  else
    copyto!(bg, bqp.c_batch)
  end
  return bg
end

function NLPModels.cons!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
  bx::AbstractMatrix{T}, bc::AbstractMatrix{T},
) where {T, S, M1, M2 <: AbstractSparseOperator, MT}
  mul!(bc, bqp.data.A, bx)
  return bc
end

function NLPModels.jac_structure!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: AbstractSparseOperator}
  fill_structure!(operator_sparse_matrix(bqp.data.A), jrows, jcols)
  return jrows, jcols
end

function NLPModels.hess_structure!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
) where {T, S, M1 <: AbstractSparseOperator, M2}
  fill_structure!(operator_sparse_matrix(bqp.data.H), hrows, hcols)
  return hrows, hcols
end

function NLPModels.jac_coord!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
) where {T, S, M1, M2 <: AbstractSparseOperator}
  bjvals .= operator_sparse_matrix(bqp.data.A).nzVal
  return bjvals
end

function NLPModels.hess_coord!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) where {T, S, M1 <: AbstractSparseOperator, M2}
  H = operator_sparse_matrix(bqp.data.H)
  nnzh = nnz(H)
  nnzh == 0 && return bhvals
  mul!(bhvals, H.nzVal, bobj_weight')
  return bhvals
end

function NLPModels.hprod!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
  bx::AbstractMatrix{T}, by::AbstractMatrix{T}, bv::AbstractMatrix{T},
  bobj_weight::AbstractVector{T}, bHv::AbstractMatrix{T},
) where {T, S, M1 <: AbstractSparseOperator, M2, MT}
  mul!(bHv, bqp.data.H, bv)
  bHv .*= bobj_weight'
  return bHv
end

function _expand_symmetric_coo(H::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
  rows, cols, vals = H.rows, H.cols, H.vals
  m, n = size(H)
  offdiag = findall(i -> rows[i] != cols[i], 1:length(rows))
  new_rows = vcat(rows, cols[offdiag])
  new_cols = vcat(cols, rows[offdiag])
  new_vals = vcat(vals, vals[offdiag])
  return SparseMatrixCOO(m, n, new_rows, new_cols, new_vals)
end

function Base.convert(::Type{ObjRHSBatchQuadraticModel{T, S}}, bnlp::ObjRHSBatchQuadraticModel{T}) where {T, S<:CuArray}
  nbatch = bnlp.meta.nbatch
  nvar = bnlp.meta.nvar
  ncon = bnlp.meta.ncon

  adapted = BatchQuadraticModels._adapt_to_operator(CuArray, bnlp)
  @assert adapted !== nothing
  data_gpu, c_batch_gpu, HX_gpu, AX_gpu = adapted

  MT = typeof(c_batch_gpu)
  meta_gpu = NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch, nvar;
    x0 = CuMatrix{T}(bnlp.meta.x0),
    lvar = CuMatrix{T}(bnlp.meta.lvar),
    uvar = CuMatrix{T}(bnlp.meta.uvar),
    ncon = ncon,
    lcon = CuMatrix{T}(bnlp.meta.lcon),
    ucon = CuMatrix{T}(bnlp.meta.ucon),
    nnzj = bnlp.meta.nnzj,
    nnzh = bnlp.meta.nnzh,
    islp = bnlp.meta.islp,
    name = bnlp.meta.name,
  )

  return ObjRHSBatchQuadraticModel{T, typeof(data_gpu.c), typeof(data_gpu.H), typeof(data_gpu.A), MT}(
    meta_gpu, data_gpu, c_batch_gpu, HX_gpu, AX_gpu,
  )
end

function BatchQuadraticModels._adapt_to_operator(to, bnlp::ObjRHSBatchQuadraticModel{T}) where {T}
  if !(to isa Type{<:CuArray} || to isa CUDABackend)
    return nothing
  end

  nbatch = bnlp.meta.nbatch
  nvar = bnlp.meta.nvar
  ncon = bnlp.meta.ncon

  H_orig_csr = CUSPARSE.CuSparseMatrixCSR(bnlp.data.H)
  H_full_csr = CUSPARSE.CuSparseMatrixCSR(_expand_symmetric_coo(bnlp.data.H))
  A_csr = CUSPARSE.CuSparseMatrixCSR(bnlp.data.A)

  H_op = gpu_operator(H_full_csr; symmetric = false, spmm_ncols = nbatch)
  H_op.A = H_orig_csr
  A_op = gpu_operator(A_csr; symmetric = false, spmm_ncols = nbatch)

  c_gpu = Adapt.adapt(to, bnlp.data.c)
  v_gpu = Adapt.adapt(to, bnlp.data.v)
  data_gpu = QPData(
    bnlp.data.c0,
    c_gpu,
    v_gpu,
    H_op,
    A_op,
    bnlp.data.regularize,
    bnlp.data.selected,
    bnlp.data.σ,
  )

  c_batch_gpu = Adapt.adapt(to, bnlp.c_batch)
  HX_gpu = similar(c_batch_gpu, T, nvar, nbatch)
  AX_gpu = similar(c_batch_gpu, T, ncon, nbatch)
  fill!(HX_gpu, zero(T))
  fill!(AX_gpu, zero(T))

  return data_gpu, c_batch_gpu, HX_gpu, AX_gpu
end
