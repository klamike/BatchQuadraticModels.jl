_sparse_vals(A::SparseMatrixCOO) = A.vals
_sparse_vals(A::SparseMatrixCSC) = A.nzval
_sparse_vals(A::AbstractSparseOperator) = _sparse_vals(operator_sparse_matrix(A))

function _copy_sparse_values!(vals::AbstractMatrix, A)
  vals .= _sparse_vals(A)
  return vals
end

function _weighted_sparse_values!(vals::AbstractMatrix, A, weights::AbstractVector)
  nnz(A) == 0 && return vals
  mul!(vals, _sparse_vals(A), weights')
  return vals
end

struct ObjRHSBatchQuadraticModel{T, S, M1, M2, MT} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  data::QPData{T, S, M1, M2}
  c_batch::MT
  _HX::MT
  _CX::MT
end

@inline _objrhs_matrix_supported(A) = A isa Union{SparseMatrixCOO, SparseMatrixCSC, AbstractSparseOperator}

function ObjRHSBatchQuadraticModel(
  qp::QuadraticModel{T, S, M1, M2},
  nbatch::Int;
  MT = typeof(similar(qp.data.c, T, 0, 0)),
  x0 = fill!(MT(undef, qp.meta.nvar, nbatch), zero(T)),
  lvar = fill!(MT(undef, qp.meta.nvar, nbatch), T(-Inf)),
  uvar = fill!(MT(undef, qp.meta.nvar, nbatch), T(Inf)),
  lcon = fill!(MT(undef, qp.meta.ncon, nbatch), T(-Inf)),
  ucon = fill!(MT(undef, qp.meta.ncon, nbatch), T(Inf)),
  c = _repeat_column(MT, qp.data.c, nbatch),
  name::String = "ObjRHSBatchQP",
) where {T, S, M1, M2}
  @assert _objrhs_matrix_supported(qp.data.Q) "Dense batch Hessians are not supported"
  @assert _objrhs_matrix_supported(qp.data.A) "Dense batch Jacobians are not supported"
  nvar = qp.meta.nvar
  ncon = qp.meta.ncon
  nnzj = qp.meta.nnzj
  nnzh = qp.meta.nnzh
  meta = _batch_meta(T, MT, qp.meta, nbatch; x0 = x0, lvar = lvar, uvar = uvar, lcon = lcon, ucon = ucon, nnzh = nnzh, islp = qp.meta.islp, name = name)
  _HX = MT(undef, nvar, nbatch)
  _CX = MT(undef, nvar, nbatch)
  H_op = sparse_operator(qp.data.Q; symmetric = true)
  A_op = sparse_operator(qp.data.A)
  data = QPData(
    A_op,
    qp.data.c,
    H_op;
    lcon = qp.data.lcon,
    ucon = qp.data.ucon,
    lvar = qp.data.lvar,
    uvar = qp.data.uvar,
    c0 = qp.data.c0[],
    _v = qp.data._v,
  )
  return ObjRHSBatchQuadraticModel{T, typeof(data.c), typeof(data.Q), typeof(data.A), MT}(meta, data, c, _HX, _CX)
end

function ObjRHSBatchQuadraticModel(
  qps::Vector{QP};
  name::String = "ObjRHSBatchQP",
  validate::Bool = false,
  MT = nothing,
) where {QP <: QuadraticModel{T, S, M1, M2}} where {T, S, M1, M2}
  nbatch = length(qps)
  @assert nbatch > 0 "Need at least one model"
  MT = _resolve_batch_matrix_type(first(qps), T, MT)
  setup = _objrhs_batch_setup(qps, name, MT; validate = validate, model_name = "ObjRHSBatchQuadraticModel")
  qp1 = setup.qp1
  return ObjRHSBatchQuadraticModel(
    qp1,
    setup.nbatch;
    x0 = setup.x0,
    lvar = setup.lvar,
    uvar = setup.uvar,
    lcon = setup.lcon,
    ucon = setup.ucon,
    c = setup.c,
    name = name,
    MT = MT,
  )
end

function Adapt.adapt_structure(to, bnlp::ObjRHSBatchQuadraticModel{T}) where {T}
  data_adapted = _adapt_qpdata(to, bnlp.data)
  c_batch_adapted = Adapt.adapt(to, bnlp.c_batch)
  HX_adapted = Adapt.adapt(to, bnlp._HX)
  CX_adapted = Adapt.adapt(to, bnlp._CX)
  MT = typeof(c_batch_adapted)
  meta_adapted = _adapt_batch_meta(to, bnlp.meta)

  return ObjRHSBatchQuadraticModel{T, typeof(data_adapted.c), typeof(data_adapted.Q), typeof(data_adapted.A), MT}(
    meta_adapted, data_adapted, c_batch_adapted, HX_adapted, CX_adapted,
  )
end

function NLPModels.obj!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  bs = length(bf)
  bf_mat = reshape(bf, 1, bs)
  mul!(bqp._HX, bqp.data.Q, bx)
  bqp._HX .*= T(0.5)
  bqp._HX .+= bqp.c_batch
  batch_mapreduce!(*, +, zero(T), bf_mat, bqp._HX, bx)
  bf .+= bqp.data.c0[]
  return bf
end

function obj_subset!(
  bqp::ObjRHSBatchQuadraticModel{T},
  bx::AbstractMatrix,
  bf::AbstractVector,
  roots::AbstractVector{<:Integer},
) where {T}
  na = length(roots)
  bf_mat = reshape(bf, 1, na)
  HX = view(bqp._HX, :, 1:na)
  mul!(HX, bqp.data.Q, bx)
  HX .*= T(0.5)
  ctmp = view(bqp._CX, :, 1:na)
  gather_columns!(ctmp, bqp.c_batch, roots)
  HX .+= ctmp
  batch_mapreduce!(*, +, zero(T), bf_mat, HX, bx)
  bf .+= bqp.data.c0[]
  return bf
end

function NLPModels.grad!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where {T}
    mul!(bg, bqp.data.Q, bx)
    bg .+= bqp.c_batch
  return bg
end

function grad_subset!(
  bqp::ObjRHSBatchQuadraticModel{T},
  bx::AbstractMatrix,
  bg::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) where {T}
  HX = view(bqp._HX, :, 1:length(roots))
  mul!(HX, bqp.data.Q, bx)
  gather_columns!(bg, bqp.c_batch, roots)
  bg .+= HX
  return bg
end

function NLPModels.cons!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where {T}
  mul!(bc, bqp.data.A, bx)
  return bc
end

cons_subset!(bqp::ObjRHSBatchQuadraticModel, bx::AbstractMatrix, bc::AbstractMatrix, ::AbstractVector{<:Integer}) =
  NLPModels.cons!(bqp, bx, bc)

function NLPModels.jac_structure!(
  bqp::ObjRHSBatchQuadraticModel,
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
)
  @lencheck bqp.meta.nnzj jrows jcols
  return _copy_sparse_structure!(bqp.data.A, jrows, jcols)
end

function NLPModels.jac_coord!(
  bqp::ObjRHSBatchQuadraticModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
) 
  return _copy_sparse_values!(bjvals, bqp.data.A)
end

jac_coord_subset!(bqp::ObjRHSBatchQuadraticModel, bx::AbstractMatrix, bjvals::AbstractMatrix, ::AbstractVector{<:Integer}) =
  NLPModels.jac_coord!(bqp, bx, bjvals)

function NLPModels.hess_structure!(
  bqp::ObjRHSBatchQuadraticModel,
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
)
  return _copy_sparse_structure!(bqp.data.Q, hrows, hcols)
end

function NLPModels.hess_coord!(
  bqp::ObjRHSBatchQuadraticModel,
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) 
  return _weighted_sparse_values!(bhvals, bqp.data.Q, bobj_weight)
end

hess_coord_subset!(bqp::ObjRHSBatchQuadraticModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bhvals::AbstractMatrix, ::AbstractVector{<:Integer}) =
  NLPModels.hess_coord!(bqp, bx, by, bobj_weight, bhvals)
