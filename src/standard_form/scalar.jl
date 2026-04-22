# Standard-form reformulation: min c'z + c0 s.t. Az = b, z >= 0 (+ optional (1/2)z'Qz).
# Each original variable splits into 1 or 2 standard variables per bound kind;
# each inequality constraint adds a slack variable. A StandardFormWorkspace
# caches the mapping so value updates can be pushed incrementally via
# `update_standard_form!` without rebuilding the structure.
#
# Bound kinds:
#   VAR_LB       l finite, u infinite      -> z = x - l
#   VAR_LB_UB    l, u finite, l != u       -> z = x - l, w = u - x
#   VAR_UB       l infinite, u finite      -> z = u - x
#   VAR_FREE     both infinite             -> x = z1 - z2
#   VAR_EQ       l == u                    -> eliminated
#   CON_EQ       l == u                    -> row kept, no slack
#   CON_LB       l finite, u infinite      -> row + slack,  Ax - s = l
#   CON_RANGE    both finite               -> row + slack + upper-complement
#   CON_UB       l infinite, u finite      -> row + slack,  Ax + s = u
@enum BoundKind::UInt8 BK_NONE VAR_LB VAR_LB_UB VAR_UB VAR_FREE VAR_EQ CON_EQ CON_LB CON_RANGE CON_UB

"""
    ScatterMap(base, dest, src, scale)

Linear scatter from a source vector into a destination slot:
`dest .= base; dest[dest[k]] += scale[k] * src[src[k]]` for each k. `base`
captures destination entries that don't depend on the source (slack
identity-rows, zeros for entries overwritten by the scatter).
"""
struct ScatterMap{T, VT <: AbstractVector{T}, VI <: AbstractVector{Int}}
  base::VT
  dest::VI
  src::VI
  scale::VT
end

"""
    BoundMap{T, LU, VI, VU}

Per-row/column bound metadata. `kind[i]` is the [`BoundKind`](@ref) of orig
entry `i`; `idx1`/`idx2` are the std-form slot(s) it maps to (one for `LB`,
`LB_UB` (lower), `UB`, `FREE` (positive), and both filled for `LB_UB`
(upper) and `FREE` (negative)); `l`/`u` hold the current bounds. `row` is
populated for constraint maps (std row per source constraint), empty for
variable maps. `LU` is a vector for the scalar path, a `(dim, nbatch)` matrix
for the batched path.
"""
struct BoundMap{T, LU <: Union{AbstractVector{T}, AbstractMatrix{T}}, VI <: AbstractVector{Int}, VU <: AbstractVector{BoundKind}}
  kind::VU
  idx1::VI
  idx2::VI
  l::LU
  u::LU
  row::VI
end

"""
    StandardFormLayout{T, VT, VI, VU}

One-shot output of the standard-form build: std-form dimensions, initial
primal/dual iterates, rhs, variable offset, source-to-std-slot maps
(`var_start`/`con_start`/`var_upper_row`/`con_upper_row`), and the
slack/identity-row contributions to the std-form matrix (`extra_I/J/V`).
Consumed once by [`StandardFormWorkspace`](@ref) and discarded.
"""
struct StandardFormLayout{T, VT <: AbstractVector{T}, VI <: AbstractVector{Int}, VU <: AbstractVector{BoundKind}}
  nstd::Int
  nrows::Int
  x0::VT
  y0::VT
  rhs::VT
  x_offset::VT
  var_lower::VI
  var_upper::VI
  var_start::BoundMap{T, VT, VI, VU}
  con_start::BoundMap{T, VT, VI, VU}
  var_upper_row::VI
  con_upper_row::VI
  extra_I::VI
  extra_J::VI
  extra_V::VT
end

"""
    StandardFormWorkspace{T, MA, MQ, VT, VI, VU}

Cached mapping from an orig scalar [`LinearModel`](@ref)/[`QuadraticModel`](@ref)
to its standard-form image. Reused by `update_standard_form!` to push value
changes through without rebuilding structure. `MQ = Nothing` on LP — `_apply!`
branches on `Q_ref === nothing` to skip Hessian work.
"""
struct StandardFormWorkspace{T, MA, MQ, VT <: AbstractVector{T}, VI <: AbstractVector{Int}, VU <: AbstractVector{BoundKind}}
  A_ref::MA
  A_src::VT
  A_map::ScatterMap{T, VT, VI}
  c_map::ScatterMap{T, VT, VI}
  signature::UInt
  rhs_base::VT
  x_offset::VT
  var_start::BoundMap{T, VT, VI, VU}
  con_start::BoundMap{T, VT, VI, VU}
  var_lower::VI
  var_upper::VI
  var_upper_row::VI
  con_upper_row::VI
  shift::VT
  activity::VT
  Q_ref::MQ
  Q_src::VT
  Q_map::ScatterMap{T, VT, VI}
  qx::VT
  ctmp::VT
end

Adapt.@adapt_structure StandardFormLayout
Adapt.@adapt_structure ScatterMap
Adapt.@adapt_structure BoundMap
Adapt.@adapt_structure StandardFormWorkspace

# ---------- sparse-matrix accessors (generic over CSC / COO / operators) ----

function _sparse_structure(A::Union{SparseMatrixCSC, SparseMatrixCOO})
  rows = Vector{Int}(undef, _nnz(A))
  cols = similar(rows)
  _copy_sparse_structure!(A, rows, cols)
  return rows, cols
end
_sparse_structure(A::AbstractSparseOperator) = _sparse_structure(operator_sparse_matrix(A))
_sparse_structure(A::BatchSparseOperator)    = (A.rows, A.cols)

_sparse_values(A::SparseMatrixCSC) = SparseArrays.nonzeros(A)
_sparse_values(A::SparseMatrixCOO) = A.vals
_sparse_values(A::AbstractSparseOperator) = _sparse_values(operator_sparse_matrix(A))
_sparse_values(A::BatchSparseOperator)    = A.nzvals

# ---------- structural signature (reject incompatible incremental updates) --

_structure_hash(A) = _structure_hash(operator_sparse_matrix(A))
_structure_hash(A::SparseMatrixCSC) = hash((size(A), A.colptr, A.rowval))
_structure_hash(A::SparseMatrixCOO) = hash((A.rows, A.cols, size(A)))

function _bound_type_code(l, u)
  lfin = isfinite(l)
  ufin = isfinite(u)
  if lfin
    return ufin ? (l == u ? VAR_EQ : VAR_LB_UB) : VAR_LB
  end
  return ufin ? VAR_UB : VAR_FREE
end

function _bounds_signature(seed::UInt, l::AbstractVector, u::AbstractVector)
  h = seed
  lh, uh = Array(l), Array(u)
  @inbounds for i in eachindex(lh, uh)
    h = hash(_bound_type_code(lh[i], uh[i]), h)
  end
  return h
end

function _structure_signature(model::_ScalarModel)
  data = model.data
  h = hash((size(data.A), NLPModels.get_nvar(model), NLPModels.get_ncon(model)))
  h = hash(_structure_hash(data.A), h)
  if model isa QuadraticModel
    h = hash(_structure_hash(data.Q), hash(size(data.Q), h))
  end
  h = _bounds_signature(h, model.meta.lvar, model.meta.uvar)
  h = _bounds_signature(h, model.meta.lcon, model.meta.ucon)
  return h
end

@inline function _standard_var_width(kind::BoundKind)
  return kind == VAR_FREE ? 2 : kind == BK_NONE ? 0 : 1
end

Base.@kwdef struct _Dirty
  c::Bool = false
  c0::Bool = false
  A::Bool = false
  Q::Bool = false
  var_bounds::Bool = false
  con_bounds::Bool = false
  x0::Bool = false
  y0::Bool = false
end
const _ALL_DIRTY = _Dirty(true, true, true, true, true, true, true, true)
const _NO_DIRTY  = _Dirty()

# Translate which `update_standard_form!` kwargs were actually passed into a
# `_Dirty` flag set; an all-`nothing` call (no kwargs given) means full refresh.
function _dirty_from_kwargs(c, c0, A, Q, lvar, uvar, lcon, ucon, x0, y0)
  d = _Dirty(
    c  = c  !== nothing, c0 = c0 !== nothing,
    A  = A  !== nothing, Q  = Q  !== nothing,
    var_bounds = lvar !== nothing || uvar !== nothing,
    con_bounds = lcon !== nothing || ucon !== nothing,
    x0 = x0 !== nothing, y0 = y0 !== nothing,
  )
  return d == _NO_DIRTY ? _ALL_DIRTY : d
end

# ---------- workspace-typed accessors (let the apply path stay shape-agnostic) ----

# `std.data.lcon`/`ucon` for scalar models, `std.meta.lcon`/`ucon` for batch.
_std_lcon(std::_ScalarModel) = std.data.lcon
_std_ucon(std::_ScalarModel) = std.data.ucon

# Scalar workspace stores a single shared op; copy current nzvals → ws scratch
# then scatter into std. `_scatter_through_scratch!` captures the shared
# scratch+scatter pattern used by both A and Q (and by the batch shared-op path).
@inline function _scatter_through_scratch!(dest_nzvals, map::ScatterMap, ref, scratch)
  _copy_sparse_values!(ref, scratch)
  _apply_scatter_map!(dest_nzvals, map, scratch)
end

_scatter_A!(std, ws::StandardFormWorkspace, _) =
  _scatter_through_scratch!(_sparse_values(std.data.A), ws.A_map, ws.A_ref, ws.A_src)
_scatter_Q!(std, ws::StandardFormWorkspace, _) =
  _scatter_through_scratch!(_sparse_values(std.data.Q), ws.Q_map, ws.Q_ref, ws.Q_src)

# Scalar c0 update: the std-form constant absorbs `src.c0 + c'x_offset`, plus
# the quadratic correction `x_offset'Q x_offset / 2` in the QP case. Batch.jl
# overloads both with `_coldot!` over the batch.
function _set_lp_c0!(std, ws::StandardFormWorkspace, src)
  std.data.c0[] = src.data.c0[] + dot(src.data.c, ws.x_offset)
  return
end

function _set_qp_c0!(std, ws::StandardFormWorkspace, src)
  std.data.c0[] = src.data.c0[] + dot(src.data.c, ws.x_offset) + dot(ws.qx, ws.x_offset) / 2
  return
end

# Scalar c-temp scatter: ctmp = src.c + qx, then scatter via c_map.
function _scatter_c_with_q!(std, ws::StandardFormWorkspace, src)
  ws.ctmp .= src.data.c .+ ws.qx
  _apply_scatter_map!(std.data.c, ws.c_map, ws.ctmp)
  return
end

_scatter_c!(std, ws::StandardFormWorkspace, src) =
  _apply_scatter_map!(std.data.c, ws.c_map, src.data.c)

# The unified `_apply!` / `_apply_lp_objective!` / `_apply_qp_objective!` live
# in `batch.jl` (loaded after `StandardFormBatchWorkspace` is defined) and
# dispatch over both workspace kinds via the accessors above and their batch
# counterparts.

# ---------- public build / update API ----------

"""
    std, ws = standard_form(orig::LinearModel | ::QuadraticModel)

Reformulate `orig` into standard form. Returns the standard-form model `std`
and a [`StandardFormWorkspace`](@ref) that caches the mapping so that value
updates on `orig` can be pushed through `update_standard_form!` without
rebuilding.

LP and QP share the layout build, the trivial-model check, the model wrap,
and the initial update; only the std-side data ctor and workspace builder
differ (dispatched on `orig`'s type).
"""
function standard_form(orig::_ScalarModel)
  layout = _build_standard_layout(orig)
  data = _build_standard_data(orig, layout)
  isempty(data.c) && throw(ArgumentError(
    "Standard-form reformulation eliminated all decision variables; trivial all-fixed models are not supported."))
  std = _wrap_standard_model(orig, data, layout)
  ws  = _build_standard_workspace(orig, std.data, layout)
  update_standard_form!(orig, std, ws)
  return std, ws
end

function _build_standard_data(lp::LinearModel, layout)
  A_rows, A_cols = _sparse_structure(lp.data.A)
  return _build_standard_linear_data(layout, A_rows, A_cols)
end
function _build_standard_data(qp::QuadraticModel, layout)
  A_rows, A_cols = _sparse_structure(qp.data.A)
  Q_rows, Q_cols = _sparse_structure(qp.data.Q)
  return _build_standard_quadratic_data(layout, A_rows, A_cols, Q_rows, Q_cols)
end

# Std-form model: `data` carries the reformulated A/Q/c, `layout` the fresh x0/y0.
_wrap_standard_model(orig::_ScalarModel, data::LPData, layout) =
  LinearModel(data;    x0 = layout.x0, y0 = layout.y0, minimize = orig.meta.minimize, name = orig.meta.name)
_wrap_standard_model(orig::_ScalarModel, data::QPData, layout) =
  QuadraticModel(data; x0 = layout.x0, y0 = layout.y0, minimize = orig.meta.minimize, name = orig.meta.name)


# Dispatched absorbers: scalar models hold c/c0/A/Q on `data`; batch models on
# `c_batch`/`c0_batch` and the operator field directly.
function _absorb_objective!(orig::_ScalarModel, c, c0)
  c  === nothing || copyto!(orig.data.c, c)
  c0 === nothing || (orig.data.c0[] = c0)
  return
end

function _absorb_matrices!(orig::_ScalarModel, A, Q)
  A === nothing || copyto!(_sparse_values(orig.data.A), _sparse_values(A))
  Q === nothing || copyto!(_sparse_values(orig.data.Q), _sparse_values(Q))
  return
end

# `update_standard_form!` is the unified body in `batch.jl` (Union-dispatched
# on workspace kind), defined after `StandardFormBatchWorkspace`.
