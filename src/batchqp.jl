"""
    BatchQuadraticModel{T, MT, VT, VI}

Batch quadratic model where all instances share the same sparsity structure
but may have different QP data.
"""
function _structure_arrays(A::SparseMatrixCOO)
  return Vector{Int}(A.rows), Vector{Int}(A.cols)
end

function _structure_arrays(A::SparseMatrixCSC)
  rows = Vector{Int}(undef, nnz(A))
  cols = Vector{Int}(undef, nnz(A))
  fill_structure!(A, rows, cols)
  return rows, cols
end

struct BatchQuadraticModel{T, MT, VT <: AbstractVector{T}, VI <: AbstractVector{Int}} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  c_batch::MT
  c0_batch::VT
  H_nzvals::MT
  A_nzvals::MT
  hess_rows::VI
  hess_cols::VI
  A_rows::VI
  A_cols::VI
  jac_op::BatchSparseOp
  jact_op::BatchSparseOp
  hess_op::BatchSparseOp
  _HX::MT
end

function Adapt.adapt_structure(to, bnlp::BatchQuadraticModel{T}) where {T}
  nbatch = bnlp.meta.nbatch
  nvar = bnlp.meta.nvar
  ncon = bnlp.meta.ncon

  c_batch_adapted = Adapt.adapt(to, bnlp.c_batch)
  c0_batch_adapted = Adapt.adapt(to, bnlp.c0_batch)
  jac_op_adapted = Adapt.adapt(to, bnlp.jac_op)
  jact_op_adapted = Adapt.adapt(to, bnlp.jact_op)
  hess_op_adapted = Adapt.adapt(to, bnlp.hess_op)
  H_nzvals_adapted = hess_op_adapted.nzVals
  A_nzvals_adapted = jac_op_adapted.nzVals
  hess_rows_adapted = Adapt.adapt(to, bnlp.hess_rows)
  hess_cols_adapted = Adapt.adapt(to, bnlp.hess_cols)
  A_rows_adapted = Adapt.adapt(to, bnlp.A_rows)
  A_cols_adapted = Adapt.adapt(to, bnlp.A_cols)
  HX_adapted = similar(c_batch_adapted, T, nvar, nbatch)
  fill!(HX_adapted, zero(T))

  MT = typeof(c_batch_adapted)
  VT = typeof(c0_batch_adapted)
  VI = typeof(hess_rows_adapted)

  meta_adapted = NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch,
    nvar;
    x0 = Adapt.adapt(to, bnlp.meta.x0),
    lvar = Adapt.adapt(to, bnlp.meta.lvar),
    uvar = Adapt.adapt(to, bnlp.meta.uvar),
    ncon = ncon,
    lcon = Adapt.adapt(to, bnlp.meta.lcon),
    ucon = Adapt.adapt(to, bnlp.meta.ucon),
    nnzj = bnlp.meta.nnzj,
    nnzh = bnlp.meta.nnzh,
    islp = bnlp.meta.islp,
    name = bnlp.meta.name,
  )

  return BatchQuadraticModel{T, MT, VT, VI}(
    meta_adapted,
    c_batch_adapted,
    c0_batch_adapted,
    H_nzvals_adapted,
    A_nzvals_adapted,
    hess_rows_adapted,
    hess_cols_adapted,
    A_rows_adapted,
    A_cols_adapted,
    jac_op_adapted,
    jact_op_adapted,
    hess_op_adapted,
    HX_adapted,
  )
end

function BatchQuadraticModel(
  qps::Vector{QP};
  name::String = "SameStructBatchQP",
  MT = typeof(similar(first(qps).data.c, T, 0, 0)),
) where {QP <: QuadraticModel{T}} where {T}
  nbatch = length(qps)
  @assert nbatch > 0 "Need at least one model"
  qp1 = first(qps)
  nvar = qp1.meta.nvar
  ncon = qp1.meta.ncon
  nnzj = qp1.meta.nnzj
  nnzh = qp1.meta.nnzh

  for qp in qps[2:end]
    @assert qp.meta.nvar == nvar "All models must have same nvar"
    @assert qp.meta.ncon == ncon "All models must have same ncon"
    @assert qp.meta.nnzj == nnzj "All models must have same nnzj"
    @assert qp.meta.nnzh == nnzh "All models must have same nnzh"
  end

  x0 = MT(reduce(hcat, [qp.meta.x0 for qp in qps]))
  lvar = MT(reduce(hcat, [qp.meta.lvar for qp in qps]))
  uvar = MT(reduce(hcat, [qp.meta.uvar for qp in qps]))
  lcon = MT(reduce(hcat, [qp.meta.lcon for qp in qps]))
  ucon = MT(reduce(hcat, [qp.meta.ucon for qp in qps]))

  meta = NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch,
    nvar;
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = nnzh,
    islp = qp1.meta.islp,
    name = name,
  )

  c_batch = MT(undef, nvar, nbatch)
  for (i, qp) in enumerate(qps)
    copyto!(view(c_batch, :, i), qp.data.c)
  end
  c0_batch = similar(c_batch, T, nbatch)
  copyto!(c0_batch, T[qp.data.c0 for qp in qps])

  hess_rows_vec, hess_cols_vec = _structure_arrays(qp1.data.H)
  hess_rows = Vector{Int}(hess_rows_vec)
  hess_cols = Vector{Int}(hess_cols_vec)

  A_nzvals = MT(undef, nnzj, nbatch)
  for (i, qp) in enumerate(qps)
    copyto!(view(A_nzvals, :, i), nonzeros(qp.data.A))
  end
  H_nzvals = MT(undef, nnzh, nbatch)
  for (i, qp) in enumerate(qps)
    copyto!(view(H_nzvals, :, i), nonzeros(qp.data.H))
  end

  A_rows_vec, A_cols_vec = _structure_arrays(qp1.data.A)

  jac_identity = collect(1:nnzj)
  jac_rowptr, jac_colidx = _coo_to_csr(Vector{Int}(A_rows_vec), ncon)
  jac_val_map = Vector{Int}(A_cols_vec)
  jac_op = _build_op(A_nzvals, jac_rowptr, jac_identity, jac_val_map, jac_colidx)

  jact_rowptr, jact_colidx = _coo_to_csr(Vector{Int}(A_cols_vec), nvar)
  jact_val_map = Vector{Int}(A_rows_vec)
  jact_op = _build_op(A_nzvals, jact_rowptr, copy(jac_identity), jact_val_map, jact_colidx)

  off_diag = findall(hess_rows .!= hess_cols)
  sym_scatter_rows = vcat(Vector{Int}(hess_rows), Vector{Int}(hess_cols[off_diag]))
  base_idx = collect(1:nnzh)
  sym_nz_idx = vcat(base_idx, Vector{Int}(off_diag))
  sym_gather_cols = vcat(Vector{Int}(hess_cols), Vector{Int}(hess_rows[off_diag]))

  hess_rowptr, hess_colidx = _coo_to_csr(sym_scatter_rows, nvar)
  hess_op = _build_op(H_nzvals, hess_rowptr, sym_nz_idx, sym_gather_cols, hess_colidx)

  VT = typeof(c0_batch)
  VI = typeof(hess_rows)
  _HX = fill!(MT(undef, nvar, nbatch), zero(T))

  return BatchQuadraticModel{T, MT, VT, VI}(
    meta,
    c_batch,
    c0_batch,
    H_nzvals,
    A_nzvals,
    hess_rows,
    hess_cols,
    A_rows_vec,
    A_cols_vec,
    jac_op,
    jact_op,
    hess_op,
    _HX,
  )
end

function NLPModels.obj!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  batch_spmv!(bqp._HX, bqp.hess_op, bx)
  bf .= bqp.c0_batch .+ vec(sum(bqp.c_batch .* bx, dims = 1)) .+ T(0.5) .* vec(sum(bx .* bqp._HX, dims = 1))
  return bf
end

function NLPModels.grad!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where {T}
  batch_spmv!(bg, bqp.hess_op, bx)
  bg .+= bqp.c_batch
  return bg
end

function NLPModels.cons!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where {T}
  batch_spmv!(bc, bqp.jac_op, bx)
  return bc
end

function NLPModels.jac_structure!(
  bqp::BatchQuadraticModel,
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
)
  @lencheck bqp.meta.nnzj jrows jcols
  copyto!(jrows, bqp.A_rows)
  copyto!(jcols, bqp.A_cols)
  return jrows, jcols
end

function NLPModels.jac_coord!(
  bqp::BatchQuadraticModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
)
  bjvals .= bqp.A_nzvals
  return bjvals
end

# function NLPModels.jprod!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJv::AbstractMatrix) where {T}
#   batch_spmv!(bJv, bqp.jac_op, bv)
#   return bJv
# end

# function NLPModels.jtprod!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix) where {T}
#   batch_spmv!(bJtv, bqp.jact_op, bv)
#   return bJtv
# end

function NLPModels.hess_structure!(
  bqp::BatchQuadraticModel,
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
)
  copyto!(hrows, bqp.hess_rows)
  copyto!(hcols, bqp.hess_cols)
  return hrows, hcols
end

function NLPModels.hess_coord!(
  bqp::BatchQuadraticModel{T},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) where {T}
  bhvals .= bqp.H_nzvals .* bobj_weight'
  return bhvals
end

# function NLPModels.hprod!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHv::AbstractMatrix) where {T}
#   batch_spmv!(bHv, bqp.hess_op, bv)
#   bHv .*= bobj_weight'
#   return bHv
# end
