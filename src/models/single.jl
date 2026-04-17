_nnz(A::SparseMatrixCOO) = length(A.vals)
_nnz(A) = SparseArrays.nnz(A)
const _HostIntView = SubArray{Int, 1, <:Vector{Int}, <:Tuple{UnitRange{Int}}, true}
const _HostValueView{T} = SubArray{T, 1, <:Vector{T}, <:Tuple{UnitRange{Int}}, true}
const _HostIntBuffer = Union{Vector{Int}, _HostIntView}
const _HostValueBuffer{T} = Union{Vector{T}, _HostValueView{T}}

function _copy_sparse_structure!(A::SparseMatrixCSC, rows::_HostIntBuffer, cols::_HostIntBuffer)
  @assert length(cols) == length(rows) == nnz(A)
  @inbounds for j in axes(A, 2)
    for k in A.colptr[j]:(A.colptr[j + 1] - 1)
      rows[k] = A.rowval[k]
      cols[k] = j
    end
  end
  return rows, cols
end

function _copy_sparse_values!(A::SparseMatrixCSC, vals::_HostValueBuffer{T}) where {T}
  @assert length(vals) == nnz(A)
  copyto!(vals, nonzeros(A))
  return vals
end

function _copy_sparse_structure!(A::SparseMatrixCOO, rows::_HostIntBuffer, cols::_HostIntBuffer)
  @assert length(rows) == length(cols) == length(A.vals)
  rows .= A.rows
  cols .= A.cols
  return rows, cols
end

function _copy_sparse_values!(A::SparseMatrixCOO, vals::_HostValueBuffer{T}) where {T}
  @assert length(vals) == length(A.vals)
  vals .= A.vals
  return vals
end

_copy_sparse_structure!(A::AbstractSparseOperator, rows::_HostIntBuffer, cols::_HostIntBuffer) =
  _copy_sparse_structure!(operator_sparse_matrix(A), rows, cols)

_copy_sparse_values!(A::AbstractSparseOperator, vals::_HostValueBuffer{T}) where {T} =
  _copy_sparse_values!(operator_sparse_matrix(A), vals)

_default_lcon(c, A) = fill!(similar(c, size(A, 1)), eltype(c)(-Inf))
_default_ucon(c, A) = fill!(similar(c, size(A, 1)), eltype(c)(Inf))
_default_lvar(c) = fill!(similar(c), eltype(c)(-Inf))
_default_uvar(c) = fill!(similar(c), eltype(c)(Inf))
_default_x0(c) = fill!(similar(c), zero(eltype(c)))
_default_y0(c, A) = fill!(similar(c, size(A, 1)), zero(eltype(c)))
_mul_jt!(jtv, A::SparseMatrixCSC, v) = mul!(jtv, transpose(A), v)
_mul_jt!(jtv, A::SparseMatrixCOO, v) = mul!(jtv, transpose(_csc_matrix(A)), v)
_mul_jt!(jtv, A::SparseOperator, v) = _mul_jt!(jtv, operator_sparse_matrix(A), v)

struct LPData{T, VT, M}
  A::M
  lcon::VT
  ucon::VT
  lvar::VT
  uvar::VT
  c::VT
  c0::Base.RefValue{T}
end

function LPData(
  A::M,
  c::VT;
  lcon::VT = _default_lcon(c, A),
  ucon::VT = _default_ucon(c, A),
  lvar::VT = _default_lvar(c),
  uvar::VT = _default_uvar(c),
  c0 = zero(eltype(c)),
) where {VT, M}
  T = eltype(c)
  return LPData{T, VT, M}(A, lcon, ucon, lvar, uvar, c, c0 isa Base.RefValue ? c0 : Ref{T}(c0))
end

mutable struct LinearModel{T, VT, M} <: NLPModels.AbstractNLPModel{T, VT}
  data::LPData{T, VT, M}
  meta::NLPModels.NLPModelMeta{T, VT}
  counters::NLPModels.Counters
end

function LinearModel(
  data::LPData{T, VT};
  x0::VT = _default_x0(data.c),
  y0::VT = _default_y0(data.c, data.A),
  minimize::Bool = true,
  name::String = "LinearModel",
) where {T, VT}
  isempty(data.c) && throw(ArgumentError("Trivial models with no decision variables are not supported."))
  return LinearModel(
    data,
    NLPModels.NLPModelMeta{T, VT}(
      length(data.c);
      lvar = data.lvar,
      uvar = data.uvar,
      ncon = size(data.A, 1),
      lcon = data.lcon,
      ucon = data.ucon,
      nnzj = _nnz(data.A),
      nnzh = 0,
      x0 = x0,
      y0 = y0,
      minimize = minimize,
      islp = true,
      name = name,
    ),
    NLPModels.Counters(),
  )
end

struct QPData{T, VT, MQ, MA}
  A::MA
  Q::MQ
  lcon::VT
  ucon::VT
  lvar::VT
  uvar::VT
  c::VT
  c0::Base.RefValue{T}
  _v::VT
end

function QPData(
  A::MA,
  c::VT,
  Q::MQ;
  lcon::VT = _default_lcon(c, A),
  ucon::VT = _default_ucon(c, A),
  lvar::VT = _default_lvar(c),
  uvar::VT = _default_uvar(c),
  c0 = zero(eltype(c)),
  _v::VT = similar(c),
) where {VT, MQ, MA}
  T = eltype(c)
  return QPData{T, VT, MQ, MA}(A, Q, lcon, ucon, lvar, uvar, c, c0 isa Base.RefValue ? c0 : Ref{T}(c0), _v)
end

mutable struct QuadraticModel{T, VT, MQ, MA} <: NLPModels.AbstractNLPModel{T, VT}
  data::QPData{T, VT, MQ, MA}
  meta::NLPModels.NLPModelMeta{T, VT}
  counters::NLPModels.Counters
end

function QuadraticModel(
  data::QPData{T, VT};
  x0::VT = _default_x0(data.c),
  y0::VT = _default_y0(data.c, data.A),
  minimize::Bool = true,
  name::String = "QuadraticModel",
) where {T, VT}
  isempty(data.c) && throw(ArgumentError("Trivial models with no decision variables are not supported."))
  return QuadraticModel(
    data,
    NLPModels.NLPModelMeta{T, VT}(
      length(data.c);
      lvar = data.lvar,
      uvar = data.uvar,
      ncon = size(data.A, 1),
      lcon = data.lcon,
      ucon = data.ucon,
      nnzj = _nnz(data.A),
      nnzh = _nnz(data.Q),
      x0 = x0,
      y0 = y0,
      minimize = minimize,
      islp = _nnz(data.Q) == 0,
      name = name,
    ),
    NLPModels.Counters(),
  )
end

function NLPModels.obj(lp::LinearModel, x::AbstractVector)
  NLPModels.increment!(lp, :neval_obj)
  return lp.data.c0[] + dot(lp.data.c, x)
end

function NLPModels.grad!(lp::LinearModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(lp, :neval_grad)
  copyto!(g, lp.data.c)
  return g
end

function NLPModels.jac_structure!(lp::LinearModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  return _copy_sparse_structure!(lp.data.A, rows, cols)
end

function NLPModels.jac_coord!(lp::LinearModel, x::AbstractVector, jac::AbstractVector)
  NLPModels.increment!(lp, :neval_jac)
  return _copy_sparse_values!(lp.data.A, jac)
end

function NLPModels.cons!(lp::LinearModel, x::AbstractVector, c::AbstractVector)
  NLPModels.increment!(lp, :neval_cons)
  mul!(c, sparse_operator(lp.data.A), x)
  return c
end

function NLPModels.jprod!(lp::LinearModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
  NLPModels.increment!(lp, :neval_jprod)
  mul!(jv, sparse_operator(lp.data.A), v)
  return jv
end

function NLPModels.jtprod!(lp::LinearModel, x::AbstractVector, v::AbstractVector, jtv::AbstractVector)
  NLPModels.increment!(lp, :neval_jtprod)
  _mul_jt!(jtv, lp.data.A, v)
  return jtv
end

NLPModels.hess_structure!(lp::LinearModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) = (rows, cols)
NLPModels.hess_coord!(lp::LinearModel, x::AbstractVector, hess::AbstractVector; obj_weight::Real = 1) = hess
NLPModels.hess_coord!(lp::LinearModel, x::AbstractVector, y::AbstractVector, hess::AbstractVector; obj_weight::Real = 1) = hess

function NLPModels.obj(qp::QuadraticModel, x::AbstractVector)
  NLPModels.increment!(qp, :neval_obj)
  mul!(qp.data._v, sparse_operator(qp.data.Q; symmetric = true), x)
  return qp.data.c0[] + dot(qp.data.c, x) + dot(qp.data._v, x) / 2
end

function NLPModels.grad!(qp::QuadraticModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(qp, :neval_grad)
  copyto!(g, qp.data.c)
  mul!(g, sparse_operator(qp.data.Q; symmetric = true), x, one(eltype(x)), one(eltype(x)))
  return g
end

function NLPModels.jac_structure!(qp::QuadraticModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  return _copy_sparse_structure!(qp.data.A, rows, cols)
end

function NLPModels.jac_coord!(qp::QuadraticModel, x::AbstractVector, jac::AbstractVector)
  NLPModels.increment!(qp, :neval_jac)
  return _copy_sparse_values!(qp.data.A, jac)
end

function NLPModels.cons!(qp::QuadraticModel, x::AbstractVector, c::AbstractVector)
  NLPModels.increment!(qp, :neval_cons)
  mul!(c, sparse_operator(qp.data.A), x)
  return c
end

function NLPModels.jprod!(qp::QuadraticModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
  NLPModels.increment!(qp, :neval_jprod)
  mul!(jv, sparse_operator(qp.data.A), v)
  return jv
end

function NLPModels.jtprod!(qp::QuadraticModel, x::AbstractVector, v::AbstractVector, jtv::AbstractVector)
  NLPModels.increment!(qp, :neval_jtprod)
  _mul_jt!(jtv, qp.data.A, v)
  return jtv
end

function NLPModels.hess_structure!(qp::QuadraticModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  return _copy_sparse_structure!(qp.data.Q, rows, cols)
end

function NLPModels.hess_coord!(qp::QuadraticModel, x::AbstractVector, hess::AbstractVector; obj_weight::Real = 1)
  NLPModels.increment!(qp, :neval_hess)
  _copy_sparse_values!(qp.data.Q, hess)
  hess .*= obj_weight
  return hess
end

function NLPModels.hess_coord!(
  qp::QuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  hess::AbstractVector;
  obj_weight::Real = 1,
)
  return NLPModels.hess_coord!(qp, x, hess; obj_weight = obj_weight)
end
