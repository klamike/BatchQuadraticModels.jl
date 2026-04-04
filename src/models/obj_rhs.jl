_supports_objrhs_batch_matrix(::Any) = false
_supports_objrhs_batch_matrix(::SparseMatrixCOO) = true
_supports_objrhs_batch_matrix(::SparseMatrixCSC) = true
_supports_objrhs_batch_matrix(::AbstractSparseOperator) = true

_mul_hessian!(Y, H::AbstractMatrix, X) = mul!(Y, Symmetric(H, :L), X)
_mul_hessian!(Y, H::AbstractSparseOperator, X) = mul!(Y, H, X)

_sparse_vals(A::SparseMatrixCOO) = A.vals
_sparse_vals(A::SparseMatrixCSC) = A.nzval
_sparse_vals(A::AbstractSparseOperator) = _sparse_vals(operator_sparse_matrix(A))

function _copy_sparse_structure!(
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  A::SparseMatrixCOO,
)
  rows .= A.rows
  cols .= A.cols
  return rows, cols
end

function _copy_sparse_structure!(
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  A::SparseMatrixCSC,
)
  fill_structure!(A, rows, cols)
  return rows, cols
end

function _copy_sparse_structure!(
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  A::AbstractSparseOperator,
)
  return _copy_sparse_structure!(rows, cols, operator_sparse_matrix(A))
end

function _copy_sparse_values!(vals::AbstractMatrix, A)
  vals .= _sparse_vals(A)
  return vals
end

function _weighted_sparse_values!(vals::AbstractMatrix, A, weights::AbstractVector)
  nnz(A) == 0 && return vals
  mul!(vals, _sparse_vals(A), weights')
  return vals
end

struct ObjRHSBatchQuadraticModel{T, S, M1, M2, MT} <: AbstractObjRHSBatchQuadraticModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  data::QPData{T, S, M1, M2}
  c_batch::MT
  _HX::MT
  _AX::MT
end

function _warn_objrhs_lp()
  @warn "ObjRHSBatchQuadraticModel is being used for an LP; ObjRHSBatchLinearModel is more efficient" maxlog = 1
  return nothing
end

function ObjRHSBatchQuadraticModel(
  qp::QuadraticModel{T, S, M1, M2},
  nbatch::Int;
  MT = typeof(similar(qp.data.c, T, 0, 0)),
  x0 = fill!(MT(undef, qp.meta.nvar, nbatch), zero(T)),
  lvar = fill!(MT(undef, qp.meta.nvar, nbatch), T(-Inf)),
  uvar = fill!(MT(undef, qp.meta.nvar, nbatch), T(Inf)),
  lcon = fill!(MT(undef, qp.meta.ncon, nbatch), T(-Inf)),
  ucon = fill!(MT(undef, qp.meta.ncon, nbatch), T(Inf)),
  c = copyto!(MT(undef, qp.meta.nvar, nbatch), repeat(qp.data.c, 1, nbatch)),
  name::String = "ObjRHSBatchQP",
) where {T, S, M1, M2}
  qp.meta.islp && _warn_objrhs_lp()
  @assert _supports_objrhs_batch_matrix(qp.data.H) "Dense batch Hessians are not supported"
  @assert _supports_objrhs_batch_matrix(qp.data.A) "Dense batch Jacobians are not supported"
  nvar = qp.meta.nvar
  ncon = qp.meta.ncon
  nnzj = qp.meta.nnzj
  nnzh = qp.meta.nnzh
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
    minimize = qp.meta.minimize,
    islp = qp.meta.islp,
    name = name,
  )
  _HX = MT(undef, nvar, nbatch)
  _AX = MT(undef, ncon, nbatch)
  H_op = sparse_operator(qp.data.H; symmetric = true)
  A_op = sparse_operator(qp.data.A)
  data = QPData(
    qp.data.c0,
    qp.data.c,
    qp.data.v,
    H_op,
    A_op,
    qp.data.regularize,
    qp.data.selected,
    qp.data.σ,
  )
  return ObjRHSBatchQuadraticModel{T, typeof(data.c), typeof(data.H), typeof(data.A), MT}(meta, data, c, _HX, _AX)
end

function ObjRHSBatchQuadraticModel(
  qps::Vector{QP};
  name::String = "ObjRHSBatchQP",
  validate::Bool = true,
  MT = nothing,
) where {QP <: QuadraticModel{T, S, M1, M2}} where {T, S, M1, M2}
  validate && _validate_objrhs_batch(qps, "ObjRHSBatchQuadraticModel")
  nbatch = length(qps)
  @assert nbatch > 0 "Need at least one model"
  qp1 = first(qps)
  MT = _resolve_batch_matrix_type(qp1, T, MT)
  x0 = reduce(hcat, [qp.meta.x0 for qp in qps])
  lvar = reduce(hcat, [qp.meta.lvar for qp in qps])
  uvar = reduce(hcat, [qp.meta.uvar for qp in qps])
  lcon = reduce(hcat, [qp.meta.lcon for qp in qps])
  ucon = reduce(hcat, [qp.meta.ucon for qp in qps])
  c = reduce(hcat, [qp.data.c for qp in qps])
  return ObjRHSBatchQuadraticModel(
    qp1,
    nbatch;
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    lcon = lcon,
    ucon = ucon,
    c = c,
    name = name,
    MT = MT,
  )
end

function _adapt_to_operator(to, bnlp)
  return nothing
end

function Adapt.adapt_structure(to, bnlp::ObjRHSBatchQuadraticModel{T}) where {T}
  nbatch = bnlp.meta.nbatch
  nvar = bnlp.meta.nvar
  ncon = bnlp.meta.ncon

  adapted = _adapt_to_operator(to, bnlp)  # check if backend supports preprocessed operators
  if adapted === nothing
    c_adapted = Adapt.adapt(to, bnlp.data.c)
    v_adapted = Adapt.adapt(to, bnlp.data.v)
    H_adapted = Adapt.adapt(to, bnlp.data.H)
    A_adapted = Adapt.adapt(to, bnlp.data.A)
    data_adapted = QPData(
      bnlp.data.c0,
      c_adapted,
      v_adapted,
      H_adapted,
      A_adapted,
      bnlp.data.regularize,
      bnlp.data.selected,
      bnlp.data.σ,
    )
    c_batch_adapted = Adapt.adapt(to, bnlp.c_batch)
    HX_adapted = Adapt.adapt(to, bnlp._HX)
    AX_adapted = Adapt.adapt(to, bnlp._AX)
  else
    data_adapted, c_batch_adapted, HX_adapted, AX_adapted = adapted
  end

  MT = typeof(c_batch_adapted)
  meta_adapted = NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch, nvar;
    x0 = Adapt.adapt(to, bnlp.meta.x0),
    lvar = Adapt.adapt(to, bnlp.meta.lvar),
    uvar = Adapt.adapt(to, bnlp.meta.uvar),
    ncon = ncon,
    lcon = Adapt.adapt(to, bnlp.meta.lcon),
    ucon = Adapt.adapt(to, bnlp.meta.ucon),
    nnzj = bnlp.meta.nnzj,
    nnzh = bnlp.meta.nnzh,
    minimize = bnlp.meta.minimize,
    islp = bnlp.meta.islp,
    name = bnlp.meta.name,
  )

  return ObjRHSBatchQuadraticModel{T, typeof(data_adapted.c), typeof(data_adapted.H), typeof(data_adapted.A), MT}(
    meta_adapted, data_adapted, c_batch_adapted, HX_adapted, AX_adapted,
  )
end

function NLPModels.obj!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  bs = length(bf)
  bf_mat = reshape(bf, 1, bs)
  _mul_hessian!(bqp._HX, bqp.data.H, bx)
  bqp._HX .*= T(0.5)
  bqp._HX .+= bqp.c_batch
  batch_mapreduce!(*, +, zero(T), bf_mat, bqp._HX, bx)
  bf .+= bqp.data.c0
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
  _mul_hessian!(HX, bqp.data.H, bx)
  HX .*= T(0.5)
  ctmp = similar(HX)
  gather_columns!(ctmp, bqp.c_batch, roots)
  HX .+= ctmp
  batch_mapreduce!(*, +, zero(T), bf_mat, HX, bx)
  bf .+= bqp.data.c0
  return bf
end

function NLPModels.grad!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where {T}
    _mul_hessian!(bg, bqp.data.H, bx)
    bg .+= bqp.c_batch
  return bg
end

function grad_subset!(
  bqp::ObjRHSBatchQuadraticModel{T},
  bx::AbstractMatrix,
  bg::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) where {T}
  _mul_hessian!(view(bqp._HX, :, 1:length(roots)), bqp.data.H, bx)
  gather_columns!(bg, bqp.c_batch, roots)
  bg .+= view(bqp._HX, :, 1:length(roots))
  return bg
end

function NLPModels.cons!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where {T}
  mul!(bc, bqp.data.A, bx)
  return bc
end

function cons_subset!(
  bqp::ObjRHSBatchQuadraticModel{T},
  bx::AbstractMatrix,
  bc::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) where {T}
  mul!(bc, bqp.data.A, bx)
  return bc
end

function NLPModels.jac_structure!(
  bqp::ObjRHSBatchQuadraticModel,
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
) 
  @lencheck bqp.meta.nnzj jrows jcols
  return _copy_sparse_structure!(jrows, jcols, bqp.data.A)
end

function NLPModels.jac_coord!(
  bqp::ObjRHSBatchQuadraticModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
) 
  return _copy_sparse_values!(bjvals, bqp.data.A)
end

function jac_coord_subset!(
  bqp::ObjRHSBatchQuadraticModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) 
  return _copy_sparse_values!(bjvals, bqp.data.A)
end

# function NLPModels.jprod!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJv::AbstractMatrix) where {T}
#   mul!(bJv, bqp.data.A, bv)
#   return bJv
# end

# function NLPModels.jtprod!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix) where {T}
#   mul!(bJtv, transpose(bqp.data.A), bv)
#   return bJtv
# end

function NLPModels.hess_structure!(
  bqp::ObjRHSBatchQuadraticModel,
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
) 
  return _copy_sparse_structure!(hrows, hcols, bqp.data.H)
end

function NLPModels.hess_coord!(
  bqp::ObjRHSBatchQuadraticModel,
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) 
  return _weighted_sparse_values!(bhvals, bqp.data.H, bobj_weight)
end

function hess_coord_subset!(
  bqp::ObjRHSBatchQuadraticModel,
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) 
  return _weighted_sparse_values!(bhvals, bqp.data.H, bobj_weight)
end


# function NLPModels.hprod!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHv::AbstractMatrix) where {T}
#   mul!(bHv, Symmetric(bqp.data.H, :L), bv)
#   bHv .*= bobj_weight'
#   return bHv
# end
