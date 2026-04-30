"""
    BatchQuadraticModel{T, MT, VT, AOp, QOp, CT, C0T}

Batched LP/QP model. All instances share the sparsity pattern of A and Q and
the same bound kinds; each datum picks its own storage:

- `A::AOp` / `Q::QOp` — `SparseOperator` (shared) or `BatchSparseOperator`
  (per-instance nzvals + batched operator).
- `c_batch::CT` — `AbstractVector` (shared) or `AbstractMatrix` (per-instance).
- `c0_batch::C0T` — scalar (shared) or `AbstractVector` (per-instance).

Type aliases: `ObjRHSBatchQuadraticModel` fixes A/Q to the scalar op form;
`UniformBatchQuadraticModel` fixes A/Q to the batched op form. Linear-term
kinds (c, c0) are independent of the alias.
"""
struct BatchQuadraticModel{T, MT <: AbstractMatrix{T}, VT <: AbstractVector{T},
                            AOp, QOp, CT, C0T} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  c_batch::CT
  c0_batch::C0T
  A::AOp
  Q::QOp
  _HX::MT
  _CX::MT
end

"""
    ObjRHSBatchQuadraticModel{T, MT, VT, AOp, QOp, CT, C0T}

`BatchQuadraticModel` restricted to the *ObjRHS* shape: A and Q are shared
across the batch (scalar `SparseOperator`). Only the linear terms (c, c0) and
bounds/iterates vary across instances.
"""
const ObjRHSBatchQuadraticModel{T, MT, VT,
                                 AOp <: AbstractSparseOperator,
                                 QOp <: AbstractSparseOperator,
                                 CT, C0T} =
  BatchQuadraticModel{T, MT, VT, AOp, QOp, CT, C0T}

"""
    UniformBatchQuadraticModel{T, MT, VT, AOp, QOp, CT, C0T}

`BatchQuadraticModel` restricted to the *Uniform* shape: A and Q carry
per-instance nzvals (`BatchSparseOperator`) — each instance has its own
matrix values, sharing only the common sparsity pattern.
"""
const UniformBatchQuadraticModel{T, MT, VT,
                                  AOp <: BatchSparseOperator,
                                  QOp <: BatchSparseOperator,
                                  CT, C0T} =
  BatchQuadraticModel{T, MT, VT, AOp, QOp, CT, C0T}

# Outer constructor with all type params inferred from the fields. Drives both
# the public ctors below and `Adapt.adapt_structure`.
function BatchQuadraticModel(meta, c_batch, c0_batch, A, Q, _HX, _CX)
  T  = eltype(_HX)
  MT = typeof(_HX)
  VT = typeof(similar(_HX, T, 0))
  return BatchQuadraticModel{T, MT, VT, typeof(A), typeof(Q),
                              typeof(c_batch), typeof(c0_batch)}(
    meta, c_batch, c0_batch, A, Q, _HX, _CX,
  )
end

# ---------- public constructors ----------

"""
    BatchQuadraticModel(qps::Vector{QuadraticModel};
                        shared_A = nothing, shared_Q = nothing,
                        shared_c = nothing, shared_c0 = nothing,
                        validate = false, name, MT)

Build a batched QP/LP from `qps` (length `nbatch`). All models must share
the sparsity pattern of A, Q, and bound kinds; each `shared_*` kwarg picks
per-instance (`false`) or shared (`true`) storage for that datum; `nothing`
auto-detects (shared iff all values match).
"""
function BatchQuadraticModel(qps::Vector{<:QuadraticModel{T}};
                              shared_A  = nothing,
                              shared_Q  = nothing,
                              shared_c  = nothing,
                              shared_c0 = nothing,
                              validate::Bool = false,
                              name::String   = "BatchQP",
                              MT = nothing) where {T}
  traits = _batch_traits(qps)            # also validates compatibility
  qp1    = first(qps)
  nbatch = length(qps)
  MT     = MT === nothing ? typeof(similar(qp1.data.c, T, 0, 0)) : MT

  shared_A  = shared_A  === nothing ? traits.objrhs : shared_A
  shared_Q  = shared_Q  === nothing ? traits.objrhs : shared_Q
  shared_c  = shared_c  === nothing ? _all_equal(qps, qp -> qp.data.c)    : shared_c
  shared_c0 = shared_c0 === nothing ? _all_equal(qps, qp -> qp.data.c0[]) : shared_c0
  traits.uniform || throw(ArgumentError(
    "BatchQuadraticModel requires identical sparse structure across the batch."))
  shared_A && !traits.same_A_values && throw(ArgumentError(
    "BatchQuadraticModel with `shared_A=true` requires identical A values across the batch."))
  shared_Q && !traits.same_Q_values && throw(ArgumentError(
    "BatchQuadraticModel with `shared_Q=true` requires identical Q values across the batch."))

  x0, y0, lvar, uvar, lcon, ucon = _stack_batch_bounds(MT, qps)
  c_batch  = shared_c ? copyto!(similar(MT(undef, 0, 0), T, qp1.meta.nvar), qp1.data.c) :
                        _stack_columns(MT, qps, qp -> qp.data.c)
  c0_batch = shared_c0 ? qp1.data.c0[]                : _stack_c0(qps, T)

  A = shared_A ? sparse_operator(qp1.data.A) :
        _jacobian_op(qp1, _stack_columns(MT, qps, qp -> _sparse_values(qp.data.A)))
  Q = shared_Q ? sparse_operator(qp1.data.Q; symmetric = true) :
        _hessian_op(qp1, _stack_columns(MT, qps, qp -> _sparse_values(qp.data.Q)))

  meta = _batch_meta(T, MT, qp1.meta, nbatch; x0, y0, lvar, uvar, lcon, ucon, name)
  _HX = MT(undef, qp1.meta.nvar, nbatch)
  _CX = MT(undef, qp1.meta.nvar, nbatch)
  return BatchQuadraticModel(meta, c_batch, c0_batch, A, Q, _HX, _CX)
end

# True iff `getter(qp)` is identical across every QP in `qps`. `===` first so
# shared references short-circuit the value comparison.
function _all_equal(qps, getter)
  length(qps) < 2 && return true
  ref = getter(qps[1])
  for j in 2:length(qps)
    x = getter(qps[j])
    x === ref && continue
    x == ref || return false
  end
  return true
end

"""
    BatchQuadraticModel(qp::QuadraticModel, nbatch; shared_A = true, shared_Q = true, kwargs...)

Replicate a scalar QP into a batched model. `shared_A`/`shared_Q` choose
shared (one scalar op reused) vs per-instance (nzvals replicated `nbatch`
times) storage for the corresponding matrix.
"""
function BatchQuadraticModel(qp::QuadraticModel{T}, nbatch::Int;
                              shared_A::Bool = true,
                              shared_Q::Bool = true,
                              MT = typeof(similar(qp.data.c, T, 0, 0)),
                              x0   = _repeat_column(MT, qp.meta.x0, nbatch),
                              y0   = _repeat_column(MT, qp.meta.y0, nbatch),
                              lvar = _repeat_column(MT, qp.meta.lvar, nbatch),
                              uvar = _repeat_column(MT, qp.meta.uvar, nbatch),
                              lcon = _repeat_column(MT, qp.meta.lcon, nbatch),
                              ucon = _repeat_column(MT, qp.meta.ucon, nbatch),
                              c    = _repeat_column(MT, qp.data.c, nbatch),
                              c0   = fill!(similar(qp.data.c, T, nbatch), qp.data.c0[]),
                              name::String = "BatchQP") where {T}
  meta = _batch_meta(T, MT, qp.meta, nbatch; x0, y0, lvar, uvar, lcon, ucon, name)
  A = shared_A ? sparse_operator(qp.data.A) :
        _jacobian_op(qp, _repeat_column(MT, _sparse_values(qp.data.A), nbatch))
  Q = shared_Q ? sparse_operator(qp.data.Q; symmetric = true) :
        _hessian_op(qp, _repeat_column(MT, _sparse_values(qp.data.Q), nbatch))
  _HX = MT(undef, qp.meta.nvar, nbatch)
  _CX = MT(undef, qp.meta.nvar, nbatch)
  return BatchQuadraticModel(meta, c, c0, A, Q, _HX, _CX)
end

"""
    ObjRHSBatchQuadraticModel(qp, nbatch; kwargs...)

Alias-like constructor for [`BatchQuadraticModel`] with `shared_A = shared_Q = true`.
"""
ObjRHSBatchQuadraticModel(qp::QuadraticModel, nbatch::Int; name = "ObjRHSBatchQP", kwargs...) =
  BatchQuadraticModel(qp, nbatch; shared_A = true, shared_Q = true, name, kwargs...)

function ObjRHSBatchQuadraticModel(qps::Vector{<:QuadraticModel};
                                    name::String   = "ObjRHSBatchQP",
                                    validate::Bool = false, MT = nothing)
  # `validate=true` flows through to the BQM ctor, which (since
  # `shared_A=shared_Q=true`) checks both `traits.uniform` and `traits.objrhs`.
  return BatchQuadraticModel(qps; shared_A = true, shared_Q = true, validate, name, MT)
end

function Adapt.adapt_structure(to, bqp::BatchQuadraticModel{T}) where {T}
  meta     = _adapt_batch_meta(to, bqp.meta)
  c_batch  = Adapt.adapt(to, bqp.c_batch)
  c0_batch = Adapt.adapt(to, bqp.c0_batch)
  A        = Adapt.adapt(to, bqp.A)
  Q        = Adapt.adapt(to, bqp.Q)
  _HX      = Adapt.adapt(to, bqp._HX)
  _CX      = Adapt.adapt(to, bqp._CX)
  return BatchQuadraticModel(meta, c_batch, c0_batch, A, Q, _HX, _CX)
end

# ---------- NLPModels API ----------

function NLPModels.obj!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  mul!(bqp._HX, bqp.Q, bx, T(0.5), zero(T))
  bqp._HX .+= bqp.c_batch
  batch_mapreduce!(*, +, zero(T), reshape(bf, 1, length(bf)), bqp._HX, bx)
  bf .+= bqp.c0_batch
  return bf
end

function NLPModels.grad!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where {T}
  mul!(bg, bqp.Q, bx)
  bg .+= bqp.c_batch
  return bg
end

function NLPModels.cons!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where {T}
  mul!(bc, bqp.A, bx)
  return bc
end

function NLPModels.jac_structure!(bqp::BatchQuadraticModel, jrows::AbstractVector{<:Integer}, jcols::AbstractVector{<:Integer})
  @lencheck bqp.meta.nnzj jrows jcols
  return _copy_sparse_structure!(bqp.A, jrows, jcols)
end

function NLPModels.hess_structure!(bqp::BatchQuadraticModel, hrows::AbstractVector{<:Integer}, hcols::AbstractVector{<:Integer})
  return _copy_sparse_structure!(bqp.Q, hrows, hcols)
end

function NLPModels.jac_coord!(bqp::BatchQuadraticModel, bx::AbstractMatrix, bjvals::AbstractMatrix)
  return _copy_sparse_values!(bjvals, bqp.A)
end

function NLPModels.hess_coord!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bhvals::AbstractMatrix) where {T}
  return _weighted_sparse_values!(bhvals, bqp.Q, bobj_weight)
end
