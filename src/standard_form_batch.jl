# Batched standard-form presolve for `ObjRHSBatchQuadraticModel`. All instances
# in the batch share the sparsity pattern of A (and Q) and bound kinds. Per-
# instance values — c, lcon/ucon, lvar/uvar, x0, y0, c0 — are stored column-wise.
#
# The std form shares Astd/Qstd structure and values across the batch (because
# orig A/Q values are shared in ObjRHSBatchQuadraticModel). What varies per
# instance is the standard-form c, rhs, c0 shift, and initial iterate.

# ---------- types ----------

# Per-instance bound metadata: l/u vary per column; kind/idx/row shared.
struct BatchBoundMap{T, MT <: AbstractMatrix{T}, VI <: AbstractVector{Int}, VU <: AbstractVector{BoundKind}}
  kind::VU
  idx1::VI
  idx2::VI
  l::MT
  u::MT
  row::VI
end

Adapt.@adapt_structure BatchBoundMap

# Batched variant of `StandardFormWorkspace`. Per-instance scratch fields are
# `MT` (matrix of shape `(dim, nbatch)`); shared structural fields reuse the
# scalar types. `c0_batch::VT` tracks the per-instance constant objective
# offset added by the presolve.
struct BatchStandardFormWorkspace{T, MA, MQ, VT <: AbstractVector{T}, MT <: AbstractMatrix{T}, VI <: AbstractVector{Int}, VU <: AbstractVector{BoundKind}}
  nbatch::Int
  A_ref::MA
  A_src::VT
  A_map::ScatterMap{T, VT, VI}
  c_map::ScatterMap{T, VT, VI}
  signature::UInt
  rhs_base::MT
  x_offset::MT
  var_start::BatchBoundMap{T, MT, VI, VU}
  con_start::BatchBoundMap{T, MT, VI, VU}
  var_lower::VI
  var_upper::VI
  var_upper_row::VI
  con_upper_row::VI
  shift::MT
  activity::MT
  Q_ref::MQ
  Q_src::VT
  Q_map::ScatterMap{T, VT, VI}
  qx::MT
  ctmp::MT
  c0_batch::VT
end

Adapt.@adapt_structure BatchStandardFormWorkspace

# ---------- per-column kernels ----------

function _batch_update_x_offset!(x_offset::AbstractMatrix{T}, meta::BatchBoundMap{T}) where {T}
  @inbounds for j in axes(x_offset, 2)
    for i in eachindex(meta.kind)
      kind = meta.kind[i]
      if kind == VAR_LB || kind == VAR_LB_UB
        x_offset[i, j] = meta.l[i, j]
      elseif kind == VAR_UB
        x_offset[i, j] = meta.u[i, j]
      elseif kind == VAR_FREE
        x_offset[i, j] = zero(T)
      else
        x_offset[i, j] = meta.l[i, j]
      end
    end
  end
  return x_offset
end

function _batch_update_rhs_base!(
  rhs_base::AbstractMatrix{T},
  var_start::BatchBoundMap{T}, con_start::BatchBoundMap{T},
  var_upper_row::AbstractVector{<:Integer}, con_upper_row::AbstractVector{<:Integer},
) where {T}
  @inbounds for j in axes(rhs_base, 2)
    for i in eachindex(con_start.row)
      row = con_start.row[i]
      row == 0 && continue
      li = con_start.l[i, j]
      rhs_base[row, j] = isfinite(li) ? li : con_start.u[i, j]
    end
    for i in eachindex(var_upper_row)
      row = var_upper_row[i]
      row > 0 && (rhs_base[row, j] = var_start.u[i, j] - var_start.l[i, j])
    end
    for i in eachindex(con_upper_row)
      row = con_upper_row[i]
      row > 0 && (rhs_base[row, j] = con_start.u[i, j] - con_start.l[i, j])
    end
  end
  return rhs_base
end

function _batch_apply_rhs_shift!(rhs::AbstractMatrix{T}, rhs_base::AbstractMatrix{T}, shift::AbstractMatrix{T}, rows::AbstractVector{<:Integer}) where {T}
  copyto!(rhs, rhs_base)
  @inbounds for j in axes(rhs, 2)
    for i in eachindex(rows)
      row = rows[i]
      row > 0 && (rhs[row, j] -= shift[i, j])
    end
  end
  return rhs
end

function _batch_update_var_start!(xstd::AbstractMatrix{T}, xsrc::AbstractMatrix{T}, meta::BatchBoundMap{T}) where {T}
  fill!(xstd, zero(T))
  @inbounds for j in axes(xstd, 2)
    for i in eachindex(meta.kind)
      kind = meta.kind[i]
      if kind == VAR_LB
        xstd[meta.idx1[i], j] = xsrc[i, j] - meta.l[i, j]
      elseif kind == VAR_LB_UB
        xstd[meta.idx1[i], j] = xsrc[i, j] - meta.l[i, j]
        xstd[meta.idx2[i], j] = meta.u[i, j] - xsrc[i, j]
      elseif kind == VAR_UB
        xstd[meta.idx1[i], j] = meta.u[i, j] - xsrc[i, j]
      elseif kind == VAR_FREE
        xi = xsrc[i, j]
        xstd[meta.idx1[i], j] = max(xi, zero(T))
        xstd[meta.idx2[i], j] = max(-xi, zero(T))
      end
    end
  end
  return xstd
end

function _batch_update_constraint_start!(xstd::AbstractMatrix{T}, activity::AbstractMatrix{T}, meta::BatchBoundMap{T}) where {T}
  @inbounds for j in axes(xstd, 2)
    for i in eachindex(meta.kind)
      kind = meta.kind[i]
      if kind == CON_LB
        xstd[meta.idx1[i], j] = activity[i, j] - meta.l[i, j]
      elseif kind == CON_RANGE
        xstd[meta.idx1[i], j] = activity[i, j] - meta.l[i, j]
        xstd[meta.idx2[i], j] = meta.u[i, j] - activity[i, j]
      elseif kind == CON_UB
        xstd[meta.idx1[i], j] = meta.u[i, j] - activity[i, j]
      end
    end
  end
  return xstd
end

function _batch_update_dual_start!(ystd::AbstractMatrix{T}, ysrc::AbstractMatrix{T}, rows::AbstractVector{<:Integer}) where {T}
  fill!(ystd, zero(T))
  @inbounds for j in axes(ystd, 2)
    for i in eachindex(rows)
      row = rows[i]
      row > 0 && (ystd[row, j] = ysrc[i, j])
    end
  end
  return ystd
end

# Batched scatter: `dest[:, j] = map.base + sum_k scale[k] * src[src_idx[k], j]`.
function _batch_apply_scatter_map!(dest::AbstractMatrix{T}, map::ScatterMap{T}, src::AbstractMatrix{T}) where {T}
  # `map.base` is (nstd,), broadcasts to each column.
  dest .= map.base
  @inbounds for j in axes(dest, 2)
    for k in eachindex(map.dest)
      dest[map.dest[k], j] += map.scale[k] * src[map.src[k], j]
    end
  end
  return dest
end

# Batched dot products per column: `out[j] = sum_i a[i, j] * b[i, j]`.
function _batch_coldot!(out::AbstractVector{T}, a::AbstractMatrix{T}, b::AbstractMatrix{T}) where {T}
  @inbounds for j in eachindex(out)
    s = zero(T)
    @simd for i in axes(a, 1)
      s += a[i, j] * b[i, j]
    end
    out[j] = s
  end
  return out
end

# Per-column symmetric SpMV for CSC Q (fallback); AbstractSparseOperator
# dispatches directly.
_batch_mul_sparse_symmetric!(dest::AbstractMatrix, Q::SparseMatrixCSC, x::AbstractMatrix) =
  mul!(dest, Symmetric(Q, :L), x)
_batch_mul_sparse_symmetric!(dest::AbstractMatrix, Q, x::AbstractMatrix) = mul!(dest, Q, x)

# ---------- build ----------

# Signature per instance: shared structure + per-column bound kinds. We check
# each column against the reference (column 1) to ensure the batch is uniform.
function _batch_structure_signature(bqp::ObjRHSBatchQuadraticModel)
  data = bqp.data
  h = hash((size(data.A), NLPModels.get_nvar(bqp), NLPModels.get_ncon(bqp)))
  h = hash(_structure_hash(data.A), h)
  if hasproperty(data, :Q)
    h = hash(size(data.Q), h)
    h = hash(_structure_hash(data.Q), h)
  end
  # Use column 1's bounds for the signature (all columns must match).
  lvar = Array(@view bqp.meta.lvar[:, 1])
  uvar = Array(@view bqp.meta.uvar[:, 1])
  lcon = Array(@view bqp.meta.lcon[:, 1])
  ucon = Array(@view bqp.meta.ucon[:, 1])
  h = _bounds_signature(h, lvar, uvar)
  h = _bounds_signature(h, lcon, ucon)
  return h
end

function _validate_uniform_bound_kinds!(bqp::ObjRHSBatchQuadraticModel)
  lvar = Array(bqp.meta.lvar)
  uvar = Array(bqp.meta.uvar)
  lcon = Array(bqp.meta.lcon)
  ucon = Array(bqp.meta.ucon)
  nbatch = bqp.meta.nbatch
  @inbounds for i in axes(lvar, 1)
    k_ref = _bound_type_code(lvar[i, 1], uvar[i, 1])
    for j in 2:nbatch
      k = _bound_type_code(lvar[i, j], uvar[i, j])
      k == k_ref || throw(ArgumentError(
        "Variable $i has bound kind $k in batch $j but $k_ref in batch 1; " *
        "batch standard-form requires uniform bound kinds.",
      ))
    end
  end
  @inbounds for i in axes(lcon, 1)
    k_ref = _bound_type_code(lcon[i, 1], ucon[i, 1])
    for j in 2:nbatch
      k = _bound_type_code(lcon[i, j], ucon[i, j])
      k == k_ref || throw(ArgumentError(
        "Constraint $i has bound kind $k in batch $j but $k_ref in batch 1; " *
        "batch standard-form requires uniform bound kinds.",
      ))
    end
  end
  return bqp
end

# Build the std form for a batch. Uses instance 1 for the structural layout,
# then fills per-instance values into column-shaped scratch.
function standard_form(bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT}) where {T, S, M1, M2, MT}
  _validate_uniform_bound_kinds!(bqp)
  nbatch = bqp.meta.nbatch

  # Build layout from column 1 (representative instance).
  ref_qp = _extract_batch_instance(bqp, 1)
  layout = _build_standard_layout(ref_qp)
  A_rows, A_cols = _source_structure(bqp.data.A)
  has_q = hasproperty(bqp.data, :Q) && bqp.data.Q !== nothing

  if has_q
    Q_rows, Q_cols = _source_structure(bqp.data.Q)
    std_data = _build_standard_quadratic_data(layout, A_rows, A_cols, Q_rows, Q_cols)
  else
    std_data = _build_standard_linear_data(layout, A_rows, A_cols)
  end
  isempty(std_data.c) && throw(ArgumentError(
    "Standard-form reformulation eliminated all decision variables; trivial all-fixed models are not supported.",
  ))

  # Wrap the scalar LP/QP std data as a single QuadraticModel, then reuse the
  # scalar workspace builder to get the shared scatter maps.
  std_single = has_q ?
    QuadraticModel(std_data; x0 = layout.x0, y0 = layout.y0, minimize = bqp.meta.minimize, name = bqp.meta.name) :
    LinearModel(std_data; x0 = layout.x0, y0 = layout.y0, minimize = bqp.meta.minimize, name = bqp.meta.name)
  ws_single = has_q ?
    _build_quadratic_workspace(ref_qp, std_single.data, layout) :
    _build_linear_workspace(ref_qp, std_single.data.A, layout)

  # Promote the shared scatter maps / kind arrays into a batched workspace and
  # wrap std_data.A/Q in an `ObjRHSBatchQuadraticModel` for the std batch.
  std_batch = _wrap_std_as_batch(std_single, nbatch, MT)
  ws_batch = _build_batch_workspace(bqp, ws_single, layout, nbatch, MT)
  update_standard_form!(bqp, std_batch, ws_batch)
  return std_batch, ws_batch
end

# Materialize a single-instance QuadraticModel view from batch column `j`.
function _extract_batch_instance(bqp::ObjRHSBatchQuadraticModel{T}, j::Int) where {T}
  data = bqp.data
  has_q = hasproperty(data, :Q) && data.Q !== nothing
  c_col = Array(@view bqp.c_batch[:, j])
  lvar = Array(@view bqp.meta.lvar[:, j])
  uvar = Array(@view bqp.meta.uvar[:, j])
  lcon = Array(@view bqp.meta.lcon[:, j])
  ucon = Array(@view bqp.meta.ucon[:, j])
  x0 = Array(@view bqp.meta.x0[:, j])
  y0 = Array(@view bqp.meta.y0[:, j])
  # Rebuild scalar A/Q as CPU SparseMatrixCSC so `_source_structure` and
  # `_csc_nz_index` work directly against the shared operator.
  A = _sparse_matrix_from(data.A)
  if has_q
    Q = _sparse_matrix_from(data.Q)
    qp = QuadraticModel(
      QPData(A, c_col, Q; lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar, c0 = data.c0[]);
      x0 = x0, y0 = y0, minimize = bqp.meta.minimize, name = string(bqp.meta.name, "_col", j),
    )
    return qp
  else
    return LinearModel(
      LPData(A, c_col; lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar, c0 = data.c0[]);
      x0 = x0, y0 = y0, minimize = bqp.meta.minimize, name = string(bqp.meta.name, "_col", j),
    )
  end
end

_sparse_matrix_from(A::SparseMatrixCSC) = A
_sparse_matrix_from(A::SparseMatrixCOO) = SparseMatrixCSC(A)
_sparse_matrix_from(A) = _sparse_matrix_from(operator_sparse_matrix(A))

# Wrap a scalar std LP/QP (with sparse CSC matrices) as an
# `ObjRHSBatchQuadraticModel` with per-instance columns for c/bounds/x0/y0.
function _wrap_std_as_batch(std_single::Union{LinearModel{T}, QuadraticModel{T}}, nbatch::Int, ::Type{MT}) where {T, MT}
  data = std_single.data
  has_q = hasproperty(data, :Q) && data.Q !== nothing
  # The std bounds (lvar = 0, uvar = Inf, lcon = ucon = rhs) are per-instance
  # for rhs, shared for lvar/uvar.
  nstd = NLPModels.get_nvar(std_single)
  nrows = NLPModels.get_ncon(std_single)
  # ObjRHSBatchQuadraticModel expects batch matrices on host matching MT.
  x0 = _repeat_column(MT, std_single.meta.x0, nbatch)
  y0 = _repeat_column(MT, std_single.meta.y0, nbatch)
  lvar = fill!(MT(undef, nstd, nbatch), zero(T))
  uvar = fill!(MT(undef, nstd, nbatch), T(Inf))
  lcon = _repeat_column(MT, data.lcon, nbatch)
  ucon = _repeat_column(MT, data.ucon, nbatch)
  c = _repeat_column(MT, data.c, nbatch)
  qp_for_batch = has_q ?
    QuadraticModel(data) :
    # Promote LPData to a QPData with zero Q so ObjRHSBatchQuadraticModel can
    # accept it; ObjRHSBatchLinearModel === ObjRHSBatchQuadraticModel alias.
    QuadraticModel(QPData(data.A, data.c, _zero_like(data.A); lcon = data.lcon, ucon = data.ucon, lvar = data.lvar, uvar = data.uvar, c0 = data.c0[]))
  return ObjRHSBatchQuadraticModel(qp_for_batch, nbatch; MT = MT, x0 = x0, lvar = lvar, uvar = uvar, lcon = lcon, ucon = ucon, c = c, name = std_single.meta.name)
end

function _zero_like(A::SparseMatrixCSC{T}) where {T}
  return sparse(Int[], Int[], T[], size(A, 2), size(A, 2))
end

# Build the batched workspace by inflating the scalar workspace's bound and
# scratch vectors into columns.
function _build_batch_workspace(bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT}, ws_single::StandardFormWorkspace{T}, layout::StandardFormLayout{T}, nbatch::Int, ::Type{MT_in}) where {T, S, M1, M2, MT, MT_in}
  n = NLPModels.get_nvar(bqp)
  m = NLPModels.get_ncon(bqp)
  nstd = layout.nstd
  nrows = layout.nrows

  # Fill batched bound metadata from the orig batch (columns of bqp.meta).
  var_start = BatchBoundMap{T, MT, typeof(ws_single.var_start.idx1), typeof(ws_single.var_start.kind)}(
    ws_single.var_start.kind,
    ws_single.var_start.idx1,
    ws_single.var_start.idx2,
    convert(MT, copy(bqp.meta.lvar)),
    convert(MT, copy(bqp.meta.uvar)),
    ws_single.var_start.row,
  )
  con_start = BatchBoundMap{T, MT, typeof(ws_single.con_start.idx1), typeof(ws_single.con_start.kind)}(
    ws_single.con_start.kind,
    ws_single.con_start.idx1,
    ws_single.con_start.idx2,
    convert(MT, copy(bqp.meta.lcon)),
    convert(MT, copy(bqp.meta.ucon)),
    ws_single.con_start.row,
  )

  VT = typeof(similar(bqp.meta.lvar, T, 0))
  c0_batch = fill!(VT(undef, nbatch), zero(T))

  has_q = ws_single.Q_ref !== nothing
  Q_ref = ws_single.Q_ref
  Q_src = ws_single.Q_src
  Q_map = ws_single.Q_map
  qx = has_q ? fill!(MT(undef, n, nbatch), zero(T)) : fill!(MT(undef, 0, nbatch), zero(T))
  ctmp = has_q ? fill!(MT(undef, n, nbatch), zero(T)) : fill!(MT(undef, 0, nbatch), zero(T))

  return BatchStandardFormWorkspace(
    nbatch,
    ws_single.A_ref,
    ws_single.A_src,
    ws_single.A_map,
    ws_single.c_map,
    _batch_structure_signature(bqp),
    fill!(MT(undef, nrows, nbatch), zero(T)),
    fill!(MT(undef, n, nbatch), zero(T)),
    var_start,
    con_start,
    ws_single.var_lower,
    ws_single.var_upper,
    ws_single.var_upper_row,
    ws_single.con_upper_row,
    fill!(MT(undef, m, nbatch), zero(T)),
    fill!(MT(undef, m, nbatch), zero(T)),
    Q_ref,
    Q_src,
    Q_map,
    qx,
    ctmp,
    c0_batch,
  )
end

# ---------- update ----------

const _BATCH_ALL_DIRTY = _Dirty(true, true, true, true, true, true, true, true)

function _batch_apply!(std::ObjRHSBatchQuadraticModel{T}, ws::BatchStandardFormWorkspace{T}, src::ObjRHSBatchQuadraticModel{T}, d::_Dirty) where {T}
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
    _batch_update_x_offset!(ws.x_offset, ws.var_start)
  end
  if d.con_bounds
    copyto!(ws.con_start.l, src.meta.lcon)
    copyto!(ws.con_start.u, src.meta.ucon)
  end

  # A values are shared across the batch (ObjRHS contract), so scatter once
  # into the shared std A nzvals.
  if d.A
    _copy_sparse_values!(ws.A_ref, ws.A_src)
    _apply_scatter_map!(_sparse_nzvals(std.data.A), ws.A_map, ws.A_src)
  end

  if rhs_base_dirty
    _batch_update_rhs_base!(ws.rhs_base, ws.var_start, ws.con_start, ws.var_upper_row, ws.con_upper_row)
  end
  if shift_dirty
    mul!(ws.shift, ws.A_ref, ws.x_offset)
  end
  if rhs_dirty
    _batch_apply_rhs_shift!(std.meta.lcon, ws.rhs_base, ws.shift, ws.con_start.row)
    copyto!(std.meta.ucon, std.meta.lcon)
  end

  if x0_var_dirty
    _batch_update_var_start!(std.meta.x0, src.meta.x0, ws.var_start)
  end
  if activity_dirty
    mul!(ws.activity, ws.A_ref, src.meta.x0)
  end
  if x0_slack_dirty
    _batch_update_constraint_start!(std.meta.x0, ws.activity, ws.con_start)
  end
  if d.y0 || d.con_bounds
    _batch_update_dual_start!(std.meta.y0, src.meta.y0, ws.con_start.row)
  end

  if ws.Q_ref === nothing
    _batch_apply_lp_objective!(std, ws, src, d)
  else
    _batch_apply_qp_objective!(std, ws, src, d, qx_dirty)
  end
  return std
end

function _batch_apply_lp_objective!(std::ObjRHSBatchQuadraticModel{T}, ws, src, d::_Dirty) where {T}
  if d.c
    _batch_apply_scatter_map!(std.c_batch, ws.c_map, src.c_batch)
  end
  if d.c0 || d.c || d.var_bounds
    # c0_std[j] = src.c0 + dot(src.c[:, j], x_offset[:, j])
    _batch_coldot!(ws.c0_batch, src.c_batch, ws.x_offset)
    ws.c0_batch .+= src.data.c0[]
  end
  return
end

function _batch_apply_qp_objective!(std::ObjRHSBatchQuadraticModel{T}, ws, src, d::_Dirty, qx_dirty::Bool) where {T}
  if d.Q
    _copy_sparse_values!(ws.Q_ref, ws.Q_src)
    _apply_scatter_map!(_sparse_nzvals(std.data.Q), ws.Q_map, ws.Q_src)
  end
  if qx_dirty
    _batch_mul_sparse_symmetric!(ws.qx, ws.Q_ref, ws.x_offset)
  end
  if d.c || qx_dirty
    copyto!(ws.ctmp, src.c_batch)
    ws.ctmp .+= ws.qx
    _batch_apply_scatter_map!(std.c_batch, ws.c_map, ws.ctmp)
  end
  if d.c0 || d.c || qx_dirty
    # c0_std[j] = src.c0 + dot(src.c[:, j], x_offset[:, j]) + dot(qx[:, j], x_offset[:, j]) / 2
    _batch_coldot!(ws.c0_batch, src.c_batch, ws.x_offset)
    tmp = similar(ws.c0_batch)
    _batch_coldot!(tmp, ws.qx, ws.x_offset)
    ws.c0_batch .+= tmp ./ 2
    ws.c0_batch .+= src.data.c0[]
  end
  return
end

"""
    update_standard_form!(orig::ObjRHSBatchQuadraticModel, std::ObjRHSBatchQuadraticModel,
                          ws::BatchStandardFormWorkspace;
                          c, c0, A, Q, lvar, uvar, lcon, ucon, x0, y0)

Batched variant of [`update_standard_form!`](@ref). Per-instance inputs
(`c`, `lvar`, `uvar`, `lcon`, `ucon`, `x0`, `y0`, `c0`) are matrices of shape
`(dim, nbatch)` (or a vector of length `nbatch` for `c0`). Shared inputs
(`A`, `Q`) copy nonzero values into the shared orig operators.

The per-instance constant objective offset from the presolve is written to
`ws.c0_batch`; add it to the std objective when comparing to the original.

Returns `std`.
"""
function update_standard_form!(orig::ObjRHSBatchQuadraticModel, std::ObjRHSBatchQuadraticModel, ws::BatchStandardFormWorkspace;
                               c = nothing, c0 = nothing, A = nothing, Q = nothing,
                               lvar = nothing, uvar = nothing, lcon = nothing, ucon = nothing,
                               x0 = nothing, y0 = nothing)
  c      === nothing || copyto!(orig.c_batch, c)
  c0     === nothing || (orig.data.c0[] = c0 isa Number ? c0 : first(c0))  # ObjRHS has scalar c0
  A      === nothing || copyto!(_sparse_nzvals(orig.data.A), _sparse_nzvals(A))
  Q      === nothing || copyto!(_sparse_nzvals(orig.data.Q), _sparse_nzvals(Q))
  lvar   === nothing || copyto!(orig.meta.lvar, lvar)
  uvar   === nothing || copyto!(orig.meta.uvar, uvar)
  lcon   === nothing || copyto!(orig.meta.lcon, lcon)
  ucon   === nothing || copyto!(orig.meta.ucon, ucon)
  x0     === nothing || copyto!(orig.meta.x0, x0)
  y0     === nothing || copyto!(orig.meta.y0, y0)
  var_bounds_changed = lvar !== nothing || uvar !== nothing
  con_bounds_changed = lcon !== nothing || ucon !== nothing
  any_given = c !== nothing || c0 !== nothing || A !== nothing || Q !== nothing ||
              var_bounds_changed || con_bounds_changed || x0 !== nothing || y0 !== nothing
  if (var_bounds_changed || con_bounds_changed) && _batch_structure_signature(orig) != ws.signature
    throw(ArgumentError(
      "update_standard_form! cannot absorb changes that alter bound kinds or the sparsity pattern; rebuild with standard_form(orig)",
    ))
  end
  d = any_given ? _Dirty(
    c          = c    !== nothing,
    c0         = c0   !== nothing,
    A          = A    !== nothing,
    Q          = Q    !== nothing,
    var_bounds = var_bounds_changed,
    con_bounds = con_bounds_changed,
    x0         = x0   !== nothing,
    y0         = y0   !== nothing,
  ) : _BATCH_ALL_DIRTY
  _batch_apply!(std, ws, orig, d)
  return std
end

# ---------- recovery ----------

function _batch_recover_primal_apply!(x::AbstractMatrix{T}, kind, idx1, idx2, z::AbstractMatrix{T}) where {T}
  @inbounds for j in axes(x, 2)
    for i in axes(x, 1)
      k = kind[i]
      if k == VAR_LB || k == VAR_LB_UB
        x[i, j] += z[idx1[i], j]
      elseif k == VAR_UB
        x[i, j] -= z[idx1[i], j]
      elseif k == VAR_FREE
        x[i, j] += z[idx1[i], j] - z[idx2[i], j]
      end
    end
  end
  return x
end

"""
    recover_primal!(x, ws::BatchStandardFormWorkspace, z)

Batched primal recovery. `x`, `z` are matrices of shape `(norig, nbatch)` and
`(nstd, nbatch)` respectively.
"""
function recover_primal!(x::AbstractMatrix, ws::BatchStandardFormWorkspace, z::AbstractMatrix)
  copyto!(x, ws.x_offset)
  _batch_recover_primal_apply!(x, ws.var_start.kind, ws.var_start.idx1, ws.var_start.idx2, z)
  return x
end

recover_primal(ws::BatchStandardFormWorkspace{T, MA, MQ, VT, MT}, z::AbstractMatrix) where {T, MA, MQ, VT, MT} =
  recover_primal!(similar(ws.x_offset), ws, z)

function _batch_scatter_multipliers!(zl::AbstractMatrix{T}, zu::AbstractMatrix{T}, var_lower, var_upper, zstd::AbstractMatrix{T}) where {T}
  fill!(zl, zero(T))
  fill!(zu, zero(T))
  @inbounds for j in axes(zl, 2)
    for i in axes(zl, 1)
      li, ui = var_lower[i], var_upper[i]
      li > 0 && (zl[i, j] = zstd[li, j])
      ui > 0 && (zu[i, j] = zstd[ui, j])
    end
  end
  return zl, zu
end

"""
    recover_variable_multipliers!(zl, zu, ws::BatchStandardFormWorkspace, zstd)

Batched variable-multiplier recovery.
"""
function recover_variable_multipliers!(zl::AbstractMatrix, zu::AbstractMatrix, ws::BatchStandardFormWorkspace, zstd::AbstractMatrix)
  _batch_scatter_multipliers!(zl, zu, ws.var_lower, ws.var_upper, zstd)
  return zl, zu
end

function _batch_gather_dual!(mult::AbstractMatrix{T}, rows, y::AbstractMatrix{T}) where {T}
  fill!(mult, zero(T))
  @inbounds for j in axes(mult, 2)
    for i in axes(mult, 1)
      row = rows[i]
      row > 0 && (mult[i, j] = y[row, j])
    end
  end
  return mult
end
