"""
    LPData(A, c; lcon, ucon, lvar, uvar, c0)

Linear-program data: `min c'x + c0  s.t.  lcon ≤ Ax ≤ ucon,  lvar ≤ x ≤ uvar`.
`A` should be `SparseMatrixCOO`; it's wrapped via [`sparse_operator`](@ref).
`c0` is stored as a length-1 vector matching `c`'s backend so device-resident
updates (`_set_c0!`) stay sync-free on GPU.
"""
struct LPData{T, VT, M}
  A::M
  lcon::VT
  ucon::VT
  lvar::VT
  uvar::VT
  c::VT
  c0::VT
end

# Wrap c0 (Number, RefValue, or AbstractVector) into a length-1 vector
# matching `c`'s backend (Vector for CPU, CuVector for GPU, etc.).
_wrap_c0(c::AbstractVector{T}, c0::AbstractVector) where {T} = c0
_wrap_c0(c::AbstractVector{T}, c0::Base.RefValue) where {T} = (v = similar(c, T, 1); v[1] = c0[]; v)
_wrap_c0(c::AbstractVector{T}, c0::Number) where {T} = (v = similar(c, T, 1); v .= T(c0); v)

function LPData(A, c::VT;
                lcon::VT = fill!(similar(c, size(A, 1)), eltype(c)(-Inf)),
                ucon::VT = fill!(similar(c, size(A, 1)), eltype(c)( Inf)),
                lvar::VT = fill!(similar(c),             eltype(c)(-Inf)),
                uvar::VT = fill!(similar(c),             eltype(c)( Inf)),
                c0 = zero(eltype(c))) where {VT}
  A_op = sparse_operator(A)
  return LPData{eltype(c), VT, typeof(A_op)}(A_op, lcon, ucon, lvar, uvar, c, _wrap_c0(c, c0))
end

"""
    LinearModel(data::LPData; x0, y0, minimize, name)

NLPModel wrapping an [`LPData`](@ref). Hessian API is a no-op.
"""
mutable struct LinearModel{T, VT, M} <: NLPModels.AbstractNLPModel{T, VT}
  data::LPData{T, VT, M}
  meta::NLPModels.NLPModelMeta{T, VT}
  counters::NLPModels.Counters
end

"""
    QPData(A, c, Q; lcon, ucon, lvar, uvar, c0, _v)

Quadratic-program data: `min c'x + (1/2)x'Qx + c0` with bounds/constraints as in
[`LPData`](@ref). `A`/`Q` should be `SparseMatrixCOO`. `_v` is a reusable `Qx`
workspace.
"""
struct QPData{T, VT, W, MQ, MA}
  A::MA
  Q::MQ
  lcon::VT
  ucon::VT
  lvar::VT
  uvar::VT
  c::VT
  c0::VT
  _v::W
end

function QPData(A, c::VT, Q;
                lcon::VT = fill!(similar(c, size(A, 1)), eltype(c)(-Inf)),
                ucon::VT = fill!(similar(c, size(A, 1)), eltype(c)( Inf)),
                lvar::VT = fill!(similar(c),             eltype(c)(-Inf)),
                uvar::VT = fill!(similar(c),             eltype(c)( Inf)),
                c0 = zero(eltype(c)),
                _v = similar(c)) where {VT}
  A_op = sparse_operator(A)
  Q_op = sparse_operator(Q; symmetric = true)
  return QPData{eltype(c), VT, typeof(_v), typeof(Q_op), typeof(A_op)}(
    A_op, Q_op, lcon, ucon, lvar, uvar, c, _wrap_c0(c, c0), _v)
end

"""
    QuadraticModel(data::QPData; x0, y0, minimize, name)

NLPModel wrapping a [`QPData`](@ref). Exposes the full NLPModels API.
"""
mutable struct QuadraticModel{T, VT, W, MQ, MA} <: NLPModels.AbstractNLPModel{T, VT}
  data::QPData{T, VT, W, MQ, MA}
  meta::NLPModels.NLPModelMeta{T, VT}
  counters::NLPModels.Counters
end

const _ScalarModel = Union{LinearModel, QuadraticModel}

function _scalar_meta(data; nnzh, islp, x0, y0, minimize, name)
  T = eltype(data.c); VT = typeof(data.c)
  isempty(data.c) && throw(ArgumentError("Trivial models with no decision variables are not supported."))
  return NLPModels.NLPModelMeta{T, VT}(
    length(data.c);
    lvar = data.lvar, uvar = data.uvar,
    ncon = size(data.A, 1), lcon = data.lcon, ucon = data.ucon,
    nnzj = nnz(data.A), nnzh, x0, y0, minimize, islp, name,
  )
end

function LinearModel(data::LPData{T, VT};
                     x0::VT = fill!(similar(data.c), zero(T)),
                     y0::VT = fill!(similar(data.c, size(data.A, 1)), zero(T)),
                     minimize::Bool = true,
                     name::String = "LinearModel") where {T, VT}
  meta = _scalar_meta(data; nnzh = 0, islp = true, x0, y0, minimize, name)
  return LinearModel(data, meta, NLPModels.Counters())
end

function QuadraticModel(data::QPData{T, VT};
                        x0::VT = fill!(similar(data.c), zero(T)),
                        y0::VT = fill!(similar(data.c, size(data.A, 1)), zero(T)),
                        minimize::Bool = true,
                        name::String = "QuadraticModel") where {T, VT}
  nnzh = nnz(data.Q)
  meta = _scalar_meta(data; nnzh, islp = nnzh == 0, x0, y0, minimize, name)
  return QuadraticModel(data, meta, NLPModels.Counters())
end


NLPModels.jac_structure!(m::_ScalarModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) =
  _copy_sparse_structure!(m.data.A, rows, cols)

function NLPModels.jac_coord!(m::_ScalarModel, x::AbstractVector, jac::AbstractVector)
  NLPModels.increment!(m, :neval_jac)
  return _copy_sparse_values!(m.data.A, jac)
end

function NLPModels.cons!(m::_ScalarModel, x::AbstractVector, c::AbstractVector)
  NLPModels.increment!(m, :neval_cons)
  mul!(c, m.data.A, x)
  return c
end

function NLPModels.jprod!(m::_ScalarModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
  NLPModels.increment!(m, :neval_jprod)
  mul!(jv, m.data.A, v)
  return jv
end

function NLPModels.jtprod!(m::_ScalarModel, x::AbstractVector, v::AbstractVector, jtv::AbstractVector)
  NLPModels.increment!(m, :neval_jtprod)
  _mul_jt!(jtv, m.data.A, v)
  return jtv
end


function NLPModels.obj(lp::LinearModel, x::AbstractVector)
  NLPModels.increment!(lp, :neval_obj)
  return @inbounds lp.data.c0[1] + dot(lp.data.c, x)
end

function NLPModels.grad!(lp::LinearModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(lp, :neval_grad)
  copyto!(g, lp.data.c)
  return g
end

NLPModels.hess_structure!(::LinearModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) = (rows, cols)
NLPModels.hess_coord!(::LinearModel, ::AbstractVector, hess::AbstractVector; obj_weight::Real = 1) = hess
NLPModels.hess_coord!(::LinearModel, ::AbstractVector, ::AbstractVector, hess::AbstractVector; obj_weight::Real = 1) = hess


function NLPModels.obj(qp::QuadraticModel, x::AbstractVector)
  NLPModels.increment!(qp, :neval_obj)
  mul!(qp.data._v, qp.data.Q, x)
  return @inbounds qp.data.c0[1] + dot(qp.data.c, x) + dot(qp.data._v, x) / 2
end

function NLPModels.grad!(qp::QuadraticModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(qp, :neval_grad)
  copyto!(g, qp.data.c)
  mul!(g, qp.data.Q, x, one(eltype(x)), one(eltype(x)))
  return g
end

NLPModels.hess_structure!(qp::QuadraticModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) =
  _copy_sparse_structure!(qp.data.Q, rows, cols)

function NLPModels.hess_coord!(qp::QuadraticModel, x::AbstractVector, hess::AbstractVector; obj_weight::Real = 1)
  NLPModels.increment!(qp, :neval_hess)
  _copy_sparse_values!(qp.data.Q, hess)
  hess .*= obj_weight
  return hess
end

NLPModels.hess_coord!(qp::QuadraticModel, x::AbstractVector, y::AbstractVector, hess::AbstractVector; obj_weight::Real = 1) =
  NLPModels.hess_coord!(qp, x, hess; obj_weight)
