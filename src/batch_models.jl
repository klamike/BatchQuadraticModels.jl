"""
    BatchQuadraticModel

Batched LP/QP sharing a common sparsity pattern and bound kinds. Each datum
(`A`, `Q`, `c_batch`, `c0_batch`) is either shared across instances or stored
per-instance. See aliases [`ObjRHSBatchQuadraticModel`](@ref) and
[`UniformBatchQuadraticModel`](@ref).
"""
struct BatchQuadraticModel{T, MT <: AbstractMatrix{T},
                           AOp, QOp, CT, C0T} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  c_batch::CT
  c0_batch::C0T
  A::AOp
  Q::QOp
  _HX::MT
end

const ObjRHSBatchQuadraticModel{T, MT,
                                AOp <: AbstractSparseOperator,
                                QOp <: AbstractSparseOperator, CT, C0T} =
  BatchQuadraticModel{T, MT, AOp, QOp, CT, C0T}

const UniformBatchQuadraticModel{T, MT,
                                 AOp <: BatchSparseOperator,
                                 QOp <: BatchSparseOperator, CT, C0T} =
  BatchQuadraticModel{T, MT, AOp, QOp, CT, C0T}

BatchQuadraticModel(meta, c_batch, c0_batch, A, Q, _HX) =
  BatchQuadraticModel{eltype(_HX), typeof(_HX),
                      typeof(A), typeof(Q), typeof(c_batch), typeof(c0_batch)}(
    meta, c_batch, c0_batch, A, Q, _HX)



function _batch_meta(::Type{T}, ::Type{MT}, meta, nbatch;
                     x0, y0 = fill!(MT(undef, meta.ncon, nbatch), zero(T)),
                     lvar, uvar, lcon, ucon,
                     nnzh = meta.nnzh, islp = meta.islp, name = meta.name) where {T, MT}
  return NLPModels.BatchNLPModelMeta{T, MT}(nbatch, meta.nvar;
    x0, lvar, uvar, ncon = meta.ncon, y0, lcon, ucon,
    nnzj = meta.nnzj, nnzh, minimize = meta.minimize, islp, name)
end

function _adapt_batch_meta(to, meta::NLPModels.BatchNLPModelMeta{T}) where {T}
  x0 = Adapt.adapt(to, meta.x0)
  return _batch_meta(T, typeof(x0), meta, meta.nbatch;
    x0,
    y0   = Adapt.adapt(to, meta.y0),
    lvar = Adapt.adapt(to, meta.lvar),
    uvar = Adapt.adapt(to, meta.uvar),
    lcon = Adapt.adapt(to, meta.lcon),
    ucon = Adapt.adapt(to, meta.ucon),
  )
end


"""
    BatchQuadraticModel(qps::Vector{<:QuadraticModel}; shared_A, shared_Q, shared_c, shared_c0, ...)

Build a batched QP/LP from `qps`. Each `shared_*` kwarg picks shared (`true`)
or per-instance (`false`) storage; `nothing` auto-detects (shared iff values
match across the batch). Caller must ensure all instances share sparsity
patterns and bound kinds.
"""
function BatchQuadraticModel(qps::Vector{<:QuadraticModel{T}};
                             shared_A  = nothing, shared_Q  = nothing,
                             shared_c  = nothing, shared_c0 = nothing,
                             name::String = "BatchQP",
                             MT = nothing) where {T}
  @assert !isempty(qps) "Need at least one model"
  qp1    = first(qps)
  nbatch = length(qps)
  MT     = MT === nothing ? typeof(similar(qp1.data.c, T, 0, 0)) : MT

  shared_A  = shared_A  === nothing ? _all_equal(qps, qp -> _sparse_values(qp.data.A)) : shared_A
  shared_Q  = shared_Q  === nothing ? _all_equal(qps, qp -> _sparse_values(qp.data.Q)) : shared_Q
  shared_c  = shared_c  === nothing ? _all_equal(qps, qp -> qp.data.c)        : shared_c
  shared_c0 = shared_c0 === nothing ? _all_equal(qps, qp -> @inbounds qp.data.c0[1]) : shared_c0

  x0   = _stack_columns(MT, qps, qp -> qp.meta.x0)
  y0   = _stack_columns(MT, qps, qp -> qp.meta.y0)
  lvar = _stack_columns(MT, qps, qp -> qp.meta.lvar)
  uvar = _stack_columns(MT, qps, qp -> qp.meta.uvar)
  lcon = _stack_columns(MT, qps, qp -> qp.meta.lcon)
  ucon = _stack_columns(MT, qps, qp -> qp.meta.ucon)

  c_batch  = shared_c ? copyto!(similar(MT(undef, 0, 0), T, qp1.meta.nvar), qp1.data.c) :
                        _stack_columns(MT, qps, qp -> qp.data.c)
  c0_batch = shared_c0 ? (@inbounds qp1.data.c0[1]) : T[@inbounds(qp.data.c0[1]) for qp in qps]

  A = shared_A ? sparse_operator(qp1.data.A) :
        _jacobian_op(qp1, _stack_columns(MT, qps, qp -> _sparse_values(qp.data.A)))
  Q = shared_Q ? sparse_operator(qp1.data.Q; symmetric = true) :
        _hessian_op(qp1, _stack_columns(MT, qps, qp -> _sparse_values(qp.data.Q)))

  meta = _batch_meta(T, MT, qp1.meta, nbatch; x0, y0, lvar, uvar, lcon, ucon, name)
  return BatchQuadraticModel(meta, c_batch, c0_batch, A, Q, MT(undef, qp1.meta.nvar, nbatch))
end

"""
    BatchQuadraticModel(qp::QuadraticModel, nbatch; shared_A=true, shared_Q=true, kwargs...)

Replicate a scalar QP into a batched model.
"""
function BatchQuadraticModel(qp::QuadraticModel{T}, nbatch::Int;
                             shared_A::Bool = true, shared_Q::Bool = true,
                             MT = typeof(similar(qp.data.c, T, 0, 0)),
                             x0   = _repeat_column(MT, qp.meta.x0, nbatch),
                             y0   = _repeat_column(MT, qp.meta.y0, nbatch),
                             lvar = _repeat_column(MT, qp.meta.lvar, nbatch),
                             uvar = _repeat_column(MT, qp.meta.uvar, nbatch),
                             lcon = _repeat_column(MT, qp.meta.lcon, nbatch),
                             ucon = _repeat_column(MT, qp.meta.ucon, nbatch),
                             c    = _repeat_column(MT, qp.data.c, nbatch),
                             c0   = fill!(similar(qp.data.c, T, nbatch), @inbounds qp.data.c0[1]),
                             name::String = "BatchQP") where {T}
  meta = _batch_meta(T, MT, qp.meta, nbatch; x0, y0, lvar, uvar, lcon, ucon, name)
  A = shared_A ? sparse_operator(qp.data.A) :
        _jacobian_op(qp, _repeat_column(MT, _sparse_values(qp.data.A), nbatch))
  Q = shared_Q ? sparse_operator(qp.data.Q; symmetric = true) :
        _hessian_op(qp, _repeat_column(MT, _sparse_values(qp.data.Q), nbatch))
  return BatchQuadraticModel(meta, c, c0, A, Q, MT(undef, qp.meta.nvar, nbatch))
end

ObjRHSBatchQuadraticModel(qp::QuadraticModel, nbatch::Int; name = "ObjRHSBatchQP", kwargs...) =
  BatchQuadraticModel(qp, nbatch; shared_A = true, shared_Q = true, name, kwargs...)

ObjRHSBatchQuadraticModel(qps::Vector{<:QuadraticModel}; name::String = "ObjRHSBatchQP",
                          MT = nothing) =
  BatchQuadraticModel(qps; shared_A = true, shared_Q = true, name, MT)

"""
    batch_model(qps; MT=nothing, name)

Alias for [`BatchQuadraticModel`](@ref) with default auto-detection of the
shared/per-instance split for `A` and `Q`.
"""
batch_model(qps::Vector{<:QuadraticModel}; name::String = "batch_model", MT = nothing) =
  BatchQuadraticModel(qps; name, MT)

function Adapt.adapt_structure(to, bqp::BatchQuadraticModel)
  return BatchQuadraticModel(
    _adapt_batch_meta(to, bqp.meta),
    Adapt.adapt(to, bqp.c_batch),
    Adapt.adapt(to, bqp.c0_batch),
    Adapt.adapt(to, bqp.A),
    Adapt.adapt(to, bqp.Q),
    Adapt.adapt(to, bqp._HX),
  )
end


function NLPModels.obj!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  bqp._HX .= bqp.c_batch
  mul!(bqp._HX, bqp.Q, bx, T(0.5), one(T))
  batch_mapreduce!(*, +, zero(T), reshape(bf, 1, length(bf)), bqp._HX, bx)
  bf .+= bqp.c0_batch
  return bf
end

function NLPModels.grad!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where {T}
  mul!(bg, bqp.Q, bx)
  bg .+= bqp.c_batch
  return bg
end

function NLPModels.cons!(bqp::BatchQuadraticModel, bx::AbstractMatrix, bc::AbstractMatrix)
  mul!(bc, bqp.A, bx)
  return bc
end

NLPModels.jac_structure!(bqp::BatchQuadraticModel, jrows::AbstractVector{<:Integer}, jcols::AbstractVector{<:Integer}) =
  (@lencheck bqp.meta.nnzj jrows jcols; _copy_sparse_structure!(bqp.A, jrows, jcols))

NLPModels.hess_structure!(bqp::BatchQuadraticModel, hrows::AbstractVector{<:Integer}, hcols::AbstractVector{<:Integer}) =
  _copy_sparse_structure!(bqp.Q, hrows, hcols)

NLPModels.jac_coord!(bqp::BatchQuadraticModel, ::AbstractMatrix, bjvals::AbstractMatrix) =
  _copy_sparse_values!(bjvals, bqp.A)

# Weighted Hessian: `vals[k, j] = nz[k or k,j] * weights[j]` — broadcast covers both
# the shared (`nz::Vector`) and per-instance (`nz::Matrix`) cases.
function NLPModels.hess_coord!(bqp::BatchQuadraticModel, ::AbstractMatrix, ::AbstractMatrix,
                                weights::AbstractVector, bhvals::AbstractMatrix)
  bhvals .= _sparse_values(bqp.Q) .* weights'
  return bhvals
end


@inline _gather_c!(dest::AbstractMatrix, c::AbstractVector, _) = (dest .= c; dest)
@inline _gather_c!(dest::AbstractMatrix, c::AbstractMatrix, roots) = gather_columns!(dest, c, roots)

@inline _scatter_c0_subset!(bf, c0::Number,         _)     = (bf .+= c0)
@inline _scatter_c0_subset!(bf, c0::AbstractVector, roots) = (bf .+= view(c0, roots))

@inline _subset_spmv!(out, op::AbstractSparseOperator, B, _, α, β) = mul!(out, op, B, α, β)
@inline _subset_spmv!(out, op::BatchSparseOperator,    B, roots, α, β) =
  batch_spmv_subset!(out, op, B, roots, α, β)

@inline _gather_nzvals!(dest::AbstractMatrix, op::AbstractSparseOperator, _) = (dest .= _sparse_values(op); dest)
@inline _gather_nzvals!(dest::AbstractMatrix, op::BatchSparseOperator, roots) = gather_columns!(dest, op.nzvals, roots)

# Batched NLPModels evaluations restricted to active instances `roots`. Mirror
# `NLPModels.{obj,grad,cons,jac_coord,hess_coord}!` but iterate `1:length(roots)`.
function obj_subset!(bqp::BatchQuadraticModel{T},
                     bx::AbstractMatrix, bf::AbstractVector,
                     roots::AbstractVector{<:Integer}) where {T}
  na = length(roots)
  HX = view(bqp._HX, :, 1:na)
  _gather_c!(HX, bqp.c_batch, roots)
  _subset_spmv!(HX, bqp.Q, bx, roots, T(0.5), one(T))
  batch_mapreduce!(*, +, zero(T), reshape(bf, 1, na), HX, bx)
  _scatter_c0_subset!(bf, bqp.c0_batch, roots)
  return bf
end

function grad_subset!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix,
                      roots::AbstractVector{<:Integer}) where {T}
  _gather_c!(bg, bqp.c_batch, roots)
  _subset_spmv!(bg, bqp.Q, bx, roots, one(T), one(T))
  return bg
end

cons_subset!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix,
             roots::AbstractVector{<:Integer}) where {T} =
  _subset_spmv!(bc, bqp.A, bx, roots, one(T), zero(T))

jac_coord_subset!(bqp::BatchQuadraticModel, ::AbstractMatrix, bjvals::AbstractMatrix,
                  roots::AbstractVector{<:Integer}) =
  _gather_nzvals!(bjvals, bqp.A, roots)

function hess_coord_subset!(bqp::BatchQuadraticModel, ::AbstractMatrix, ::AbstractMatrix,
                            weights::AbstractVector, bhvals::AbstractMatrix,
                            roots::AbstractVector{<:Integer})
  _gather_nzvals!(bhvals, bqp.Q, roots)
  bhvals .*= weights'
  return bhvals
end
