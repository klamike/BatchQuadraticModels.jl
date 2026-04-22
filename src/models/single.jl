# ---------- sparse-matrix copy primitives ----------
# Vector-dest `_copy_sparse_structure!`/`_copy_sparse_values!` drive scalar
# `jac_coord!`/`hess_coord!`. Matrix-dest `_copy_sparse_values!` is the batch
# counterpart; `_weighted_sparse_values!` adds the per-instance `obj_weight`
# scaling needed by batch `hess_coord!`.

_nnz(A) = SparseArrays.nnz(A)

function _copy_sparse_structure!(A::SparseMatrixCSC, rows::AbstractVector, cols::AbstractVector)
  @assert length(cols) == length(rows) == nnz(A)
  @inbounds for j in axes(A, 2)
    for k in A.colptr[j]:(A.colptr[j + 1] - 1)
      rows[k] = A.rowval[k]
      cols[k] = j
    end
  end
  return rows, cols
end

function _copy_sparse_values!(A::SparseMatrixCSC, vals::AbstractVector)
  @assert length(vals) == nnz(A)
  copyto!(vals, nonzeros(A))
  return vals
end

function _copy_sparse_structure!(A::SparseMatrixCOO, rows::AbstractVector, cols::AbstractVector)
  @assert length(rows) == length(cols) == length(A.vals)
  rows .= A.rows
  cols .= A.cols
  return rows, cols
end

function _copy_sparse_values!(A::SparseMatrixCOO, vals::AbstractVector)
  @assert length(vals) == length(A.vals)
  vals .= A.vals
  return vals
end

# Forward a `SparseOperator`/`CuSparseOperator` to its backing sparse matrix.
_copy_sparse_structure!(A::AbstractSparseOperator, rows::AbstractVector, cols::AbstractVector) =
  _copy_sparse_structure!(operator_sparse_matrix(A), rows, cols)

_copy_sparse_values!(A::AbstractSparseOperator, vals::AbstractVector) =
  _copy_sparse_values!(operator_sparse_matrix(A), vals)

# Per-instance batch op: structural rows/cols and nzvals live on the op directly.
function _copy_sparse_structure!(A::BatchSparseOperator, rows::AbstractVector, cols::AbstractVector)
  copyto!(rows, A.rows); copyto!(cols, A.cols); return rows, cols
end

# Matrix-dest value copy (batch `jac_coord!`/`hess_coord!`). Broadcasting
# handles both shapes: a vector source (shared) broadcasts to all columns, a
# matrix source (per-instance) copies elementwise.
_copy_sparse_values!(vals::AbstractMatrix, A) = (vals .= _sparse_values(A); vals)

# Weighted matrix-dest value copy (batch `hess_coord!` with per-instance
# `obj_weight`). Shared-op path uses outer product with the weight vector;
# per-instance path broadcasts the scaling column-wise.
function _weighted_sparse_values!(vals::AbstractMatrix, A::AbstractSparseOperator, weights::AbstractVector)
  nnz(A) == 0 && return vals
  mul!(vals, _sparse_values(A), weights')
  return vals
end
_weighted_sparse_values!(vals::AbstractMatrix, A::BatchSparseOperator, weights::AbstractVector) =
  (vals .= _sparse_values(A) .* weights'; vals)

# ---------- `J'v` product for `jtprod!` ----------
# `SparseMatrixCOO` has no `mul!(..., transpose(A), ...)` method; convert to
# CSC on demand. `SparseOperator` forwards through its backing source.
_mul_jt!(jtv, A::SparseMatrixCSC, v) = mul!(jtv, transpose(A), v)
_mul_jt!(jtv, A::SparseMatrixCOO, v) = mul!(jtv, transpose(_csc_matrix(A)), v)
_mul_jt!(jtv, A::SparseOperator,  v) = _mul_jt!(jtv, operator_sparse_matrix(A), v)

# ---------- ctor helpers for `LPData`/`QPData` ----------
# Default-bound helpers: shape follows `c` (nvar) or `(c, A)` (ncon).
_default_lvar(c::AbstractVector)    = fill!(similar(c),             eltype(c)(-Inf))
_default_uvar(c::AbstractVector)    = fill!(similar(c),             eltype(c)( Inf))
_default_x0(c::AbstractVector)      = fill!(similar(c),             zero(eltype(c)))
_default_lcon(c::AbstractVector, A) = fill!(similar(c, size(A, 1)), eltype(c)(-Inf))
_default_ucon(c::AbstractVector, A) = fill!(similar(c, size(A, 1)), eltype(c)( Inf))
_default_y0(c::AbstractVector, A)   = fill!(similar(c, size(A, 1)), zero(eltype(c)))

# Pass `c0` through as a `RefValue` if the caller already wrapped it; otherwise box.
@inline _as_ref(::Type{T}, c0::Base.RefValue{T}) where {T} = c0
@inline _as_ref(::Type{T}, c0)                   where {T} = Ref{T}(c0)

# ---------- LP / QP data and model types ----------

"""
    LPData(A, c; lcon, ucon, lvar, uvar, c0)

Linear-program data: `min c'x + c0  s.t.  lcon ≤ Ax ≤ ucon,  lvar ≤ x ≤ uvar`.
`A` is wrapped via [`sparse_operator`](@ref); `c0` is boxed in a `RefValue` so
scalar callers can mutate it in place.
"""
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
  A,
  c::VT;
  lcon::VT = _default_lcon(c, A),
  ucon::VT = _default_ucon(c, A),
  lvar::VT = _default_lvar(c),
  uvar::VT = _default_uvar(c),
  c0 = zero(eltype(c)),
) where {VT}
  T = eltype(c)
  A_op = sparse_operator(A)
  return LPData{T, VT, typeof(A_op)}(A_op, lcon, ucon, lvar, uvar, c, _as_ref(T, c0))
end

"""
    LinearModel(data::LPData; x0, y0, minimize, name)

`NLPModels.AbstractNLPModel` wrapping an [`LPData`](@ref). Exposes the standard
NLPModels API (`obj`, `grad!`, `cons!`, `jac_*!`, `jprod!`/`jtprod!`); the
Hessian API is a no-op.
"""
mutable struct LinearModel{T, VT, M} <: NLPModels.AbstractNLPModel{T, VT}
  data::LPData{T, VT, M}
  meta::NLPModels.NLPModelMeta{T, VT}
  counters::NLPModels.Counters
end

function _scalar_meta(::Type{T}, ::Type{VT}, data; nnzh, islp, x0, y0, minimize, name) where {T, VT}
  isempty(data.c) && throw(ArgumentError("Trivial models with no decision variables are not supported."))
  return NLPModels.NLPModelMeta{T, VT}(
    length(data.c);
    lvar = data.lvar, uvar = data.uvar,
    ncon = size(data.A, 1), lcon = data.lcon, ucon = data.ucon,
    nnzj = _nnz(data.A), nnzh, x0, y0, minimize, islp, name,
  )
end

function LinearModel(
  data::LPData{T, VT};
  x0::VT = _default_x0(data.c),
  y0::VT = _default_y0(data.c, data.A),
  minimize::Bool = true,
  name::String = "LinearModel",
) where {T, VT}
  meta = _scalar_meta(T, VT, data; nnzh = 0, islp = true, x0, y0, minimize, name)
  return LinearModel(data, meta, NLPModels.Counters())
end

"""
    QPData(A, c, Q; lcon, ucon, lvar, uvar, c0, _v)

Quadratic-program data: `min c'x + (1/2)x'Qx + c0` subject to the same bound/
constraint structure as [`LPData`](@ref). `Q` is wrapped as a symmetric
[`sparse_operator`](@ref); `_v` is a reusable `Qx` scratch vector so `obj` and
`grad!` don't allocate.
"""
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
  A,
  c::VT,
  Q;
  lcon::VT = _default_lcon(c, A),
  ucon::VT = _default_ucon(c, A),
  lvar::VT = _default_lvar(c),
  uvar::VT = _default_uvar(c),
  c0 = zero(eltype(c)),
  _v::VT = similar(c),
) where {VT}
  T = eltype(c)
  A_op = sparse_operator(A)
  Q_op = sparse_operator(Q; symmetric = true)
  return QPData{T, VT, typeof(Q_op), typeof(A_op)}(A_op, Q_op, lcon, ucon, lvar, uvar, c, _as_ref(T, c0), _v)
end

"""
    QuadraticModel(data::QPData; x0, y0, minimize, name)

`NLPModels.AbstractNLPModel` wrapping a [`QPData`](@ref). Exposes the full
NLPModels API including Hessian (`hess_structure!`/`hess_coord!`).
"""
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
  nnzh = _nnz(data.Q)
  meta = _scalar_meta(T, VT, data; nnzh, islp = nnzh == 0, x0, y0, minimize, name)
  return QuadraticModel(data, meta, NLPModels.Counters())
end

# ---------- NLPModels API: shared LP + QP (both carry `data.A`) ----------

const _ScalarModel = Union{LinearModel, QuadraticModel}

function NLPModels.jac_structure!(m::_ScalarModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  return _copy_sparse_structure!(m.data.A, rows, cols)
end

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

# ---------- NLPModels API: LP-specific ----------

function NLPModels.obj(lp::LinearModel, x::AbstractVector)
  NLPModels.increment!(lp, :neval_obj)
  return lp.data.c0[] + dot(lp.data.c, x)
end

function NLPModels.grad!(lp::LinearModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(lp, :neval_grad)
  copyto!(g, lp.data.c)
  return g
end

NLPModels.hess_structure!(lp::LinearModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) = (rows, cols)
NLPModels.hess_coord!(lp::LinearModel, x::AbstractVector, hess::AbstractVector; obj_weight::Real = 1) = hess
NLPModels.hess_coord!(lp::LinearModel, x::AbstractVector, y::AbstractVector, hess::AbstractVector; obj_weight::Real = 1) = hess

# ---------- NLPModels API: QP-specific ----------

function NLPModels.obj(qp::QuadraticModel, x::AbstractVector)
  NLPModels.increment!(qp, :neval_obj)
  mul!(qp.data._v, qp.data.Q, x)
  return qp.data.c0[] + dot(qp.data.c, x) + dot(qp.data._v, x) / 2
end

function NLPModels.grad!(qp::QuadraticModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(qp, :neval_grad)
  copyto!(g, qp.data.c)
  mul!(g, qp.data.Q, x, one(eltype(x)), one(eltype(x)))
  return g
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

NLPModels.hess_coord!(qp::QuadraticModel, x::AbstractVector, y::AbstractVector, hess::AbstractVector; obj_weight::Real = 1) =
  NLPModels.hess_coord!(qp, x, hess; obj_weight)
