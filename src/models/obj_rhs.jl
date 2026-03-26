_supports_objrhs_batch_matrix(A) = A isa SparseMatrixCOO || A isa SparseMatrixCSC

struct ObjRHSBatchQuadraticModel{T, S, M1, M2, MT} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  data::QPData{T, S, M1, M2}
  c_batch::MT
  _HX::MT
  _AX::MT
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
    islp = (nnzh == 0),
    name = name,
  )
  _HX = MT(undef, nvar, nbatch)
  _AX = MT(undef, ncon, nbatch)
  return ObjRHSBatchQuadraticModel{T, S, M1, M2, MT}(meta, qp.data, c, _HX, _AX)
end

function ObjRHSBatchQuadraticModel(
  qps::Vector{QP};
  name::String = "ObjRHSBatchQP",
  MT = typeof(similar(first(qps).data.c, T, 0, 0)),
) where {QP <: QuadraticModel{T, S, M1, M2}} where {T, S, M1, M2}
  nbatch = length(qps)
  qp1 = first(qps)
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

  adapted = _adapt_to_operator(to, bnlp)
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
    HX_adapted = similar(c_batch_adapted, T, nvar, nbatch)
    AX_adapted = similar(c_batch_adapted, T, ncon, nbatch)
    fill!(HX_adapted, zero(T))
    fill!(AX_adapted, zero(T))
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
  if !bqp.meta.islp
    mul!(bqp._HX, Symmetric(bqp.data.H, :L), bx)
    bqp._HX .*= T(0.5)
    bqp._HX .+= bqp.c_batch
    batch_mapreduce!(*, +, zero(T), bf_mat, bqp._HX, bx)
  else
    batch_mapreduce!(*, +, zero(T), bf_mat, bqp.c_batch, bx)
  end
  bf .+= bqp.data.c0
  return bf
end

function NLPModels.grad!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where {T}
    if !bqp.meta.islp
        mul!(bg, Symmetric(bqp.data.H, :L), bx)
        bg .+= bqp.c_batch
    else
        copyto!(bg, bqp.c_batch)
    end
  return bg
end

function NLPModels.cons!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where {T}
  mul!(bc, bqp.data.A, bx)
  return bc
end

function NLPModels.jac_structure!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: SparseMatrixCOO}
  @lencheck bqp.meta.nnzj jrows jcols
  jrows .= bqp.data.A.rows
  jcols .= bqp.data.A.cols
  return jrows, jcols
end

function NLPModels.jac_structure!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: SparseMatrixCSC}
  @lencheck bqp.meta.nnzj jrows jcols
  fill_structure!(bqp.data.A, jrows, jcols)
  return jrows, jcols
end

function NLPModels.jac_coord!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
) where {T, S, M1, M2 <: SparseMatrixCOO}
  bjvals .= bqp.data.A.vals
  return bjvals
end

function NLPModels.jac_coord!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
) where {T, S, M1, M2 <: SparseMatrixCSC}
  bjvals .= bqp.data.A.nzval
  return bjvals
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
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
) where {T, S, M1 <: SparseMatrixCOO, M2}
  hrows .= bqp.data.H.rows
  hcols .= bqp.data.H.cols
  return hrows, hcols
end

function NLPModels.hess_structure!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
) where {T, S, M1 <: SparseMatrixCSC, M2}
  fill_structure!(bqp.data.H, hrows, hcols)
  return hrows, hcols
end

function NLPModels.hess_coord!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) where {T, S, M1 <: SparseMatrixCOO, M2}
  mul!(bhvals, bqp.data.H.vals, bobj_weight')
  return bhvals
end

function NLPModels.hess_coord!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) where {T, S, M1 <: SparseMatrixCSC, M2}
  H = bqp.data.H
  nnzh = nnz(H)
  nnzh == 0 && return bhvals
  mul!(bhvals, H.nzVal, bobj_weight')
  return bhvals
end


# function NLPModels.hprod!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHv::AbstractMatrix) where {T}
#   mul!(bHv, Symmetric(bqp.data.H, :L), bv)
#   bHv .*= bobj_weight'
#   return bHv
# end
