function Adapt.adapt_structure(::Type{<:CuArray}, bnlp::BatchQuadraticModel{T}) where {T}
  nbatch = bnlp.meta.nbatch
  nvar = bnlp.meta.nvar
  ncon = bnlp.meta.ncon

  c_batch_gpu = CuMatrix{T}(bnlp.c_batch)
  c0_batch_gpu = CuVector{T}(bnlp.c0_batch)
  hess_rows_gpu = CuVector{Int}(bnlp.hess_rows)
  hess_cols_gpu = CuVector{Int}(bnlp.hess_cols)
  A_rows_gpu = CuVector{Int}(bnlp.A_rows)
  A_cols_gpu = CuVector{Int}(bnlp.A_cols)
  HX_gpu = CUDA.zeros(T, nvar, nbatch)
  MT = typeof(c_batch_gpu)
  jac_op_gpu = Adapt.adapt(CuArray, bnlp.jac_op)
  jact_op_gpu = Adapt.adapt(CuArray, bnlp.jact_op)
  hess_op_gpu = Adapt.adapt(CuArray, bnlp.hess_op)
  A_nzvals_gpu = jac_op_gpu.nzVals
  H_nzvals_gpu = hess_op_gpu.nzVals

  meta_gpu = NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch,
    nvar;
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

  VT = CuVector{T}
  VI = CuVector{Int}

  return BatchQuadraticModel{T, MT, VT, VI}(
    meta_gpu,
    c_batch_gpu,
    c0_batch_gpu,
    H_nzvals_gpu,
    A_nzvals_gpu,
    hess_rows_gpu,
    hess_cols_gpu,
    A_rows_gpu,
    A_cols_gpu,
    jac_op_gpu,
    jact_op_gpu,
    hess_op_gpu,
    HX_gpu,
  )
end
