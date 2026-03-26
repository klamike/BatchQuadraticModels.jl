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

function NLPModels.obj!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  H = Symmetric(bqp.data.H, :L)
  mul!(bqp._HX, H, bx)
  bf .= bqp.data.c0 .+ vec(sum(bqp.c_batch .* bx, dims = 1)) .+ T(0.5) .* vec(sum(bx .* bqp._HX, dims = 1))
  return bf
end

function NLPModels.grad!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where {T}
  mul!(bg, Symmetric(bqp.data.H, :L), bx)
  bg .+= bqp.c_batch
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

function NLPModels.jac_structure!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: Matrix}
  @lencheck bqp.meta.nnzj jrows jcols
  count = 1
  for j = 1:bqp.meta.nvar
    for i = 1:bqp.meta.ncon
      jrows[count] = i
      jcols[count] = j
      count += 1
    end
  end
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

function NLPModels.jac_coord!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
) where {T, S, M1, M2 <: Matrix}
  bjvals .= vec(bqp.data.A)
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

function NLPModels.hess_structure!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
) where {T, S, M1 <: Matrix, M2}
  count = 1
  for j = 1:bqp.meta.nvar
    for i = j:bqp.meta.nvar
      hrows[count] = i
      hcols[count] = j
      count += 1
    end
  end
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

function NLPModels.hess_coord!(
  bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) where {T, S, M1 <: Matrix, M2}
  H = bqp.data.H
  nvar = bqp.meta.nvar
  count = 1
  for j = 1:nvar
    for i = j:nvar
      @views bhvals[count, :] .= H[i, j] .* bobj_weight
      count += 1
    end
  end
  return bhvals
end

# function NLPModels.hprod!(bqp::ObjRHSBatchQuadraticModel{T}, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHv::AbstractMatrix) where {T}
#   mul!(bHv, Symmetric(bqp.data.H, :L), bv)
#   bHv .*= bobj_weight'
#   return bHv
# end
