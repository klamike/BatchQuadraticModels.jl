# Unified batched standard-form presolve for `BatchQuadraticModel`. Each A / Q
# entry on the model is one of two operator kinds (scalar `SparseOperator` or
# `BatchSparseOperator`); workspace methods dispatch on the operator type.
#
# The workspace stores both kinds uniformly: `A_ref`/`Q_ref` is the SpMV target
# (used directly via `mul!`); `A_src`/`Q_src` is a scratch nzvals buffer that
# only matters for the shared-op path (sized to nnz; length-0 for varying).

"""
    StandardFormBatchWorkspace{T, MT, VT, VI, VU, ARef, QRef}

Batched counterpart of [`StandardFormWorkspace`](@ref). `MT` is the per-instance
matrix backend (CPU `Matrix` or GPU `CuMatrix`); scratch buffers (`shift`,
`activity`, `qx`, `ctmp`, `rhs_base`) carry one column per batch instance.
`ARef`/`QRef` are the orig SpMV targets — `SparseOperator` when the op is
shared across the batch (ObjRHS shape) or `BatchSparseOperator` when each
instance has its own nzvals (Uniform shape). `QRef === Nothing` on LP so the
Hessian branch of `_apply!` is skipped.
"""
struct StandardFormBatchWorkspace{T,
    MT <: AbstractMatrix{T}, VT <: AbstractVector{T},
    VI <: AbstractVector{Int}, VU <: AbstractVector{BoundKind},
    ARef, QRef}
  A_map::ScatterMap{T, VT, VI}
  Q_map::ScatterMap{T, VT, VI}       # zero-length for LP
  c_map::ScatterMap{T, VT, VI}
  signature::UInt
  var_start::BoundMap{T, MT, VI, VU}
  con_start::BoundMap{T, MT, VI, VU}
  var_lower::VI
  var_upper::VI
  var_upper_row::VI
  con_upper_row::VI
  rhs_base::MT
  x_offset::MT
  shift::MT
  activity::MT
  qx::MT
  ctmp::MT
  c0_batch::VT
  c0_tmp::VT                         # scratch `x'Qx` accumulator (QP only)
  A_ref::ARef                        # SpMV target for A (orig op, shared or batched)
  A_src::VT                          # scratch nzvals for shared A (length 0 for varying)
  Q_ref::QRef                        # `nothing` for LP
  Q_src::VT                          # scratch nzvals for shared Q (length 0 otherwise)
end

Adapt.@adapt_structure StandardFormBatchWorkspace

# Scatter source nzvals into std nzvals. Shared sources sync the scratch
# `*_src` buffer from the live op first (so the scatter sees fresh values)
# then broadcast the scalar buffer over the batch; varying sources scatter
# their per-instance nzvals matrix directly. Both paths reuse the scalar-side
# `_scatter_through_scratch!` / `_apply_scatter_map!` helpers.
_scatter_op!(dest_nzvals, map::ScatterMap, ref, scratch, ::AbstractSparseOperator) =
  _scatter_through_scratch!(dest_nzvals, map, ref, scratch)
_scatter_op!(dest_nzvals, map::ScatterMap, _ref, _scratch, src_op::BatchSparseOperator) =
  _apply_scatter_map!(dest_nzvals, map, src_op.nzvals)

# ---------- std form building ----------

# Host-pull the bound matrices to check uniformity — gated on `--check-bounds=no`.
function _validate_uniform_bound_kinds!(bqp::BatchQuadraticModel)
  _should_verify_structure(bqp) || return
  _assert_uniform_kinds(Array(bqp.meta.lvar), Array(bqp.meta.uvar), "Variable")
  _assert_uniform_kinds(Array(bqp.meta.lcon), Array(bqp.meta.ucon), "Constraint")
  return
end

# Each batch column must produce the same `_bound_type_code` per row as col 1.
function _assert_uniform_kinds(l::AbstractMatrix, u::AbstractMatrix, label::String)
  @inbounds for i in axes(l, 1)
    k_ref = _bound_type_code(l[i, 1], u[i, 1])
    for j in 2:size(l, 2)
      _bound_type_code(l[i, j], u[i, j]) == k_ref || throw(ArgumentError(
        "$label $i has mismatched bound kind in batch $j vs batch 1; uniform bound kinds required."))
    end
  end
end

# Structural signature used to reject incompatible incremental updates.
function _structure_signature(bqp::BatchQuadraticModel)
  h = hash((bqp.meta.nvar, bqp.meta.ncon, bqp.meta.nnzj, bqp.meta.nnzh))
  _should_verify_structure(bqp) || return h
  # Host-pull sparsity indices and bound matrices only when verifying.
  A_rows, A_cols = _sparse_structure(bqp.A)
  Q_rows, Q_cols = _sparse_structure(bqp.Q)
  h = hash((Vector{Int}(A_rows), Vector{Int}(A_cols), Vector{Int}(Q_rows), Vector{Int}(Q_cols)), h)
  lvar = Array(@view bqp.meta.lvar[:, 1])
  uvar = Array(@view bqp.meta.uvar[:, 1])
  lcon = Array(@view bqp.meta.lcon[:, 1])
  ucon = Array(@view bqp.meta.ucon[:, 1])
  h = _bounds_signature(h, lvar, uvar)
  h = _bounds_signature(h, lcon, ucon)
  return h
end

# Build a scalar representative QP for the structural std-form build. Shares
# data with the batch (no host pull). c0 is zero — per-instance offsets live
# in `ws.c0_batch`.
function _representative_qp(bqp::BatchQuadraticModel{T}) where {T}
  nvar = bqp.meta.nvar; ncon = bqp.meta.ncon
  c_col = bqp.c_batch[:, 1]
  lvar = bqp.meta.lvar[:, 1]; uvar = bqp.meta.uvar[:, 1]
  lcon = bqp.meta.lcon[:, 1]; ucon = bqp.meta.ucon[:, 1]
  x0   = bqp.meta.x0[:, 1];   y0   = bqp.meta.y0[:, 1]
  A = _representative_matrix(bqp.A, c_col, ncon, nvar)
  Q = _representative_matrix(bqp.Q, c_col, nvar, nvar)
  return QuadraticModel(
    QPData(A, c_col, Q; lvar, uvar, lcon, ucon, c0 = zero(T));
    x0, y0, minimize = bqp.meta.minimize, name = string(bqp.meta.name, "_rep"))
end

# For shared entries we can reuse the existing scalar sparse op; for varying
# entries we materialize a scalar sparse matrix from column 1 of nzvals.
_representative_matrix(op::AbstractSparseOperator, _...) = op
_representative_matrix(op::BatchSparseOperator, c_col, m, n) =
  _build_scalar_sparse(c_col, op.rows, op.cols, view(op.nzvals, :, 1), m, n)

@inline _build_scalar_sparse(::AbstractVector, rows, cols, vals::AbstractVector, m, n) =
  sparse(rows, cols, vals, m, n)

# Shared scratch buffer for nzvals — sized to nnz for shared ops, length 0
# for varying (which scatter their per-instance nzvals matrix directly).
_nzvals_scratch(op::AbstractSparseOperator, ::Type{VT}) where {VT} =
  VT(undef, length(_sparse_values(op)))
_nzvals_scratch(::BatchSparseOperator, ::Type{VT}) where {VT} = VT(undef, 0)

# ---------- build standard form ----------

function standard_form(bqp::BatchQuadraticModel{T, MT, VT, AOp, QOp}) where {T, MT, VT, AOp, QOp}
  _validate_uniform_bound_kinds!(bqp)
  nbatch = bqp.meta.nbatch
  has_q = bqp.meta.nnzh > 0

  rep_qp = _representative_qp(bqp)
  std_single, ws_single = standard_form(rep_qp)
  isempty(std_single.data.c) && throw(ArgumentError(
    "Standard-form reformulation eliminated all decision variables; trivial all-fixed models are not supported."))

  std_batch = _replicate_std_as_batch(std_single, MT, nbatch;
    shared_A = AOp <: AbstractSparseOperator, shared_Q = QOp <: AbstractSparseOperator)

  var_start = _inflate_bound_map(MT, ws_single.var_start, bqp.meta.lvar, bqp.meta.uvar)
  con_start = _inflate_bound_map(MT, ws_single.con_start, bqp.meta.lcon, bqp.meta.ucon)

  n = NLPModels.get_nvar(bqp); m = NLPModels.get_ncon(bqp)
  nrows = length(ws_single.rhs_base)
  WVT = typeof(similar(bqp.meta.lvar, T, 0))
  # Scratch buffers are write-before-read on the initial `_ALL_DIRTY` update_standard_form!.
  mat(r, c) = MT(undef, r, c)

  ws = StandardFormBatchWorkspace(
    ws_single.A_map, ws_single.Q_map, ws_single.c_map,
    _structure_signature(bqp),
    var_start, con_start,
    ws_single.var_lower, ws_single.var_upper,
    ws_single.var_upper_row, ws_single.con_upper_row,
    mat(nrows, nbatch), mat(n, nbatch),
    mat(m, nbatch), mat(m, nbatch),
    mat(has_q ? n : 0, nbatch), mat(has_q ? n : 0, nbatch),
    WVT(undef, nbatch), WVT(undef, nbatch),
    bqp.A, _nzvals_scratch(bqp.A, WVT),
    has_q ? bqp.Q : nothing, has_q ? _nzvals_scratch(bqp.Q, WVT) : WVT(undef, 0),
  )
  update_standard_form!(bqp, std_batch, ws)
  return std_batch, ws
end

# Replicate the scalar std model into a batched one, preserving the orig kinds
# (Shared in → Shared out, Varying in → Varying out) so update flows keep
# their fast path. Driven by `nbatch` and the kind flags directly — no need
# for the orig bqp here.
#
# The per-instance presolve offset `c0 + c'x_offset + x_offset'Qx_offset/2`
# lives on the workspace (`ws.c0_batch`), so the batched std model itself must
# carry `c0_batch = 0` — otherwise the scalar representative's offset would be
# inherited across all instances and added twice at solution-recovery time.
function _replicate_std_as_batch(std_single, ::Type{MT}, nbatch::Int;
                                  shared_A::Bool, shared_Q::Bool) where {MT}
  data = std_single.data
  T    = eltype(data.c)
  nstd = NLPModels.get_nvar(std_single)
  qp   = _adapt_to_batch_backend(std_single, MT, nbatch)
  return BatchQuadraticModel(qp, nbatch;
    MT,
    x0   = _repeat_column(MT, std_single.meta.x0, nbatch),
    lvar = fill!(MT(undef, nstd, nbatch), zero(T)),   # std-form lower bound: 0
    uvar = fill!(MT(undef, nstd, nbatch), T(Inf)),    # std-form upper bound: +Inf
    lcon = _repeat_column(MT, data.lcon, nbatch),
    ucon = _repeat_column(MT, data.ucon, nbatch),
    c0   = fill!(similar(data.c, T, nbatch), zero(T)),
    name = std_single.meta.name, shared_A, shared_Q)
end

# No-op default; CUDA ext specializes on `MT <: CuMatrix` to rebuild sparse
# operators with `spmm_ncols = nbatch` so the batched SpMM buffer is premade.
_adapt_to_batch_backend(qp, ::Type, nbatch) = qp

# ---------- batch-side workspace accessors (mirror the scalar set) ----

# `std.meta.lcon`/`ucon` for batch models (vs `std.data.*` for scalar).
_std_lcon(std::BatchQuadraticModel) = std.meta.lcon
_std_ucon(std::BatchQuadraticModel) = std.meta.ucon

# Std A/Q nzvals destination: `_sparse_values` handles both the shared-op case
# (underlying CSR/CSC nzvals) and the varying-op case (the per-instance nzvals
# matrix carried by `BatchSparseOperator`).
_scatter_A!(std::BatchQuadraticModel, ws::StandardFormBatchWorkspace, src) =
  _scatter_op!(_sparse_values(std.A), ws.A_map, ws.A_ref, ws.A_src, src.A)
_scatter_Q!(std::BatchQuadraticModel, ws::StandardFormBatchWorkspace, src) =
  _scatter_op!(_sparse_values(std.Q), ws.Q_map, ws.Q_ref, ws.Q_src, src.Q)

# Per-instance c0 update. The LP case accumulates `src.c0_batch[j] + c'x_offset`;
# the QP case adds the `x_offset'Qx_offset / 2` correction via `ws.c0_tmp`.
function _set_lp_c0!(::BatchQuadraticModel, ws::StandardFormBatchWorkspace, src)
  _coldot!(ws.c0_batch, src.c_batch, ws.x_offset)
  ws.c0_batch .+= src.c0_batch
  return
end

function _set_qp_c0!(::BatchQuadraticModel, ws::StandardFormBatchWorkspace, src)
  _coldot!(ws.c0_batch, src.c_batch, ws.x_offset)
  _coldot!(ws.c0_tmp,   ws.qx,        ws.x_offset)
  @. ws.c0_batch += ws.c0_tmp / 2 + src.c0_batch
  return
end

function _scatter_c_with_q!(std::BatchQuadraticModel, ws::StandardFormBatchWorkspace, src)
  ws.ctmp .= src.c_batch .+ ws.qx    # shared `c::Vector` broadcasts over columns; `Matrix` c does elementwise add
  _apply_scatter_map!(std.c_batch, ws.c_map, ws.ctmp)
  return
end

_scatter_c!(std::BatchQuadraticModel, ws::StandardFormBatchWorkspace, src) =
  _apply_scatter_map!(std.c_batch, ws.c_map, src.c_batch)

# ---------- shared incremental update (Union-dispatched on workspace) ----

const _AnyStdFormWorkspace = Union{StandardFormWorkspace, StandardFormBatchWorkspace}

function _apply!(std, ws::_AnyStdFormWorkspace, src, d::_Dirty)
  rhs_base_dirty  = d.var_bounds || d.con_bounds
  shift_dirty     = d.A          || d.var_bounds
  rhs_dirty       = rhs_base_dirty || shift_dirty
  x0_var_dirty    = d.var_bounds || d.x0
  activity_dirty  = d.A          || d.x0
  x0_slack_dirty  = activity_dirty || d.con_bounds
  qx_dirty        = d.Q          || d.var_bounds

  if d.var_bounds
    copyto!(ws.var_start.l, src.meta.lvar)
    copyto!(ws.var_start.u, src.meta.uvar)
    _update_x_offset!(ws.x_offset, ws.var_start)
  end
  if d.con_bounds
    copyto!(ws.con_start.l, src.meta.lcon)
    copyto!(ws.con_start.u, src.meta.ucon)
  end

  d.A && _scatter_A!(std, ws, src)

  A = ws.A_ref   # SpMV target, used by both `shift` and `activity` updates
  if rhs_base_dirty
    _update_rhs_base!(ws.rhs_base, ws.var_start, ws.con_start, ws.var_upper_row, ws.con_upper_row)
  end
  shift_dirty && mul!(ws.shift, A, ws.x_offset)
  if rhs_dirty
    _apply_rhs_shift!(_std_lcon(std), ws.rhs_base, ws.shift, ws.con_start.row)
    copyto!(_std_ucon(std), _std_lcon(std))
  end

  x0_var_dirty   && _update_var_start!(std.meta.x0, src.meta.x0, ws.var_start)
  activity_dirty && mul!(ws.activity, A, src.meta.x0)
  x0_slack_dirty && _update_constraint_start!(std.meta.x0, ws.activity, ws.con_start)
  (d.y0 || d.con_bounds) && _update_dual_start!(std.meta.y0, src.meta.y0, ws.con_start.row)

  ws.Q_ref === nothing ? _apply_lp_objective!(std, ws, src, d) :
                         _apply_qp_objective!(std, ws, src, d, qx_dirty)
  return std
end

function _apply_lp_objective!(std, ws, src, d::_Dirty)
  d.c && _scatter_c!(std, ws, src)
  (d.c0 || d.c || d.var_bounds) && _set_lp_c0!(std, ws, src)
  return
end

function _apply_qp_objective!(std, ws, src, d::_Dirty, qx_dirty::Bool)
  d.Q      && _scatter_Q!(std, ws, src)
  qx_dirty && mul!(ws.qx, ws.Q_ref, ws.x_offset)
  c_dirty  = d.c || qx_dirty
  c_dirty            && _scatter_c_with_q!(std, ws, src)
  (d.c0 || c_dirty)  && _set_qp_c0!(std, ws, src)
  return
end

# ---------- batch-side absorbers (mirror the scalar versions in scalar.jl) ----

function _absorb_objective!(orig::BatchQuadraticModel, c, c0)
  c  === nothing || copyto!(orig.c_batch, c)
  c0 === nothing || (orig.c0_batch .= c0)  # scalar broadcasts, array copies
  return
end

# Dispatched on the entry's kind: shared scalar op (copy via `_sparse_values`)
# vs. per-instance `BatchSparseOperator` (copy directly into the nzvals matrix).
_absorb_nzvals!(op::AbstractSparseOperator, src) = copyto!(_sparse_values(op), _sparse_values(src))
_absorb_nzvals!(op::BatchSparseOperator, src)    = copyto!(op.nzvals, src)

function _absorb_matrices!(orig::BatchQuadraticModel, A, Q)
  A === nothing || _absorb_nzvals!(orig.A, A)
  Q === nothing || _absorb_nzvals!(orig.Q, Q)
  return
end

"""
    update_standard_form!(orig, std, ws; c, c0, A, Q, lvar, uvar, lcon, ucon, x0, y0)

Mutate `orig` with any provided fields (in original variable/constraint
space), then propagate the minimal set of updates to `std`. With no kwargs,
performs a full refresh from the current state of `orig`.

Sparsity patterns and bound kinds (finite ↔ infinite, `l == u`) must be
unchanged; for structural changes, rebuild via [`standard_form`](@ref).

Works on both scalar and batch standard-form workspaces.
"""
function update_standard_form!(orig, std, ws::_AnyStdFormWorkspace;
                               c = nothing, c0 = nothing, A = nothing, Q = nothing,
                               lvar = nothing, uvar = nothing, lcon = nothing, ucon = nothing,
                               x0 = nothing, y0 = nothing)
  _absorb_objective!(orig, c, c0)
  _absorb_matrices!(orig, A, Q)
  _absorb_meta!(orig.meta; lvar, uvar, lcon, ucon, x0, y0)
  if (lvar !== nothing || uvar !== nothing || lcon !== nothing || ucon !== nothing) &&
     _structure_signature(orig) != ws.signature
    throw(ArgumentError(
      "update_standard_form! cannot absorb changes that alter bound kinds " *
      "(finite ↔ infinite, l == u) or the sparsity pattern; rebuild with standard_form(orig)"))
  end
  d = _dirty_from_kwargs(c, c0, A, Q, lvar, uvar, lcon, ucon, x0, y0)
  _apply!(std, ws, orig, d)
  return std
end

# ---------- recovery (unified scalar + batch via workspace dispatch) ----------

"""
    recover_primal!(x, ws, z)

Undo the standard-form reformulation: write the original primal `x` from the
standard-form solution `z`. Vectors for `StandardFormWorkspace`, matrices for
`StandardFormBatchWorkspace`.
"""
function recover_primal!(x::AbstractVecOrMat, ws::_AnyStdFormWorkspace, z::AbstractVecOrMat)
  copyto!(x, ws.x_offset)
  _recover_primal_apply!(x, ws.var_start.kind, ws.var_start.idx1, ws.var_start.idx2, z)
  return x
end

recover_primal(ws::_AnyStdFormWorkspace, z::AbstractVecOrMat) =
  recover_primal!(similar(ws.x_offset), ws, z)

"""
    recover_variable_multipliers!(zl, zu, ws, zstd)

Scatter the standard-form non-negativity multipliers `zstd` back to the
original lower/upper bound multipliers (`zl`, `zu`). Uses
`ws.var_lower`/`ws.var_upper` to index: a non-zero `ws.var_lower[i]` identifies
the std slot carrying the original `zl[i]`, analogously for `ws.var_upper`.
Entries without a std slot receive zero.
"""
function recover_variable_multipliers!(zl::AbstractVecOrMat, zu::AbstractVecOrMat,
                                       ws::_AnyStdFormWorkspace, zstd::AbstractVecOrMat)
  _scatter_multipliers!(zl, zu, ws.var_lower, ws.var_upper, zstd)
  return zl, zu
end
