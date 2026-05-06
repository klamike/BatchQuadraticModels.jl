# Standard-form reformulation: min c'z + c0 s.t. Az = b, z >= 0 (+ optional (1/2)z'Qz).
# Bound kinds map orig vars/cons to std slots:
#   VAR_LB:    z = x - l                   VAR_LB_UB: z = x - l, w = u - x
#   VAR_UB:    z = u - x                   VAR_FREE:  x = z1 - z2
#   VAR_EQ:    eliminated                  CON_EQ:    no slack
#   CON_LB:    Ax - s = l                  CON_RANGE: slack + upper-complement
#   CON_UB:    Ax + s = u

@inline _standard_var_width(kind::BoundKind) =
  kind == VAR_FREE ? 2 : kind == BK_NONE ? 0 : 1

@inline function _foreach_standard_var(f, vs::BoundMap, idx::Int)
  T = eltype(vs.l)
  k = vs.kind[idx]
  if k == VAR_LB || k == VAR_LB_UB
    f(vs.idx1[idx], one(T))
  elseif k == VAR_UB
    f(vs.idx1[idx], -one(T))
  elseif k == VAR_FREE
    f(vs.idx1[idx], one(T))
    f(vs.idx2[idx], -one(T))
  end
  return nothing
end

_host_vector(v::Vector) = v
_host_vector(v::AbstractVector) = collect(v)

function _build_standard_layout(model)
  T = eltype(model.data.c)
  n = NLPModels.get_nvar(model); m = NLPModels.get_ncon(model)
  lvar = _host_vector(model.meta.lvar); uvar = _host_vector(model.meta.uvar)
  lcon = _host_vector(model.meta.lcon); ucon = _host_vector(model.meta.ucon)
  x0_src = _host_vector(model.meta.x0); y0_src = _host_vector(model.meta.y0)

  # Variable layout: orig var i → 0/1/2 std slots based on bound kind.
  var_kind = fill(BK_NONE, n)
  var_idx1 = zeros(Int, n); var_idx2 = zeros(Int, n)
  var_lower = zeros(Int, n); var_upper = zeros(Int, n)
  x_offset = zeros(T, n)
  x0_std = T[]
  var_upper_pairs = Tuple{Int,Int,Int}[]  # (orig_i, z_idx, w_idx)

  nstd = 0
  @inbounds for i in 1:n
    li, ui = lvar[i], uvar[i]
    if li == ui
      x_offset[i] = li
    elseif isfinite(li)
      nstd += 1
      x_offset[i] = li; var_lower[i] = nstd; var_idx1[i] = nstd
      push!(x0_std, x0_src[i] - li)
      if isfinite(ui)
        nstd += 1
        var_kind[i] = VAR_LB_UB; var_upper[i] = nstd; var_idx2[i] = nstd
        push!(x0_std, ui - x0_src[i])
        push!(var_upper_pairs, (i, var_lower[i], var_upper[i]))
      else
        var_kind[i] = VAR_LB
      end
    elseif isfinite(ui)
      nstd += 1
      x_offset[i] = ui
      var_kind[i] = VAR_UB; var_upper[i] = nstd; var_idx1[i] = nstd
      push!(x0_std, ui - x0_src[i])
    else
      nstd += 1
      var_kind[i] = VAR_FREE; var_idx1[i] = nstd
      push!(x0_std, max(x0_src[i], zero(T)))
      nstd += 1; var_idx2[i] = nstd
      push!(x0_std, max(-x0_src[i], zero(T)))
    end
  end

  # Constraint layout: each kept row becomes one primary std row + optional slack.
  con_kind = fill(BK_NONE, m)
  con_idx1 = zeros(Int, m); con_idx2 = zeros(Int, m)
  constraint_rows = zeros(Int, m)
  rhs = T[]; y0_std = T[]
  I = Int[]; J = Int[]; V = T[]
  con_upper_pairs = Tuple{Int,Int,Int}[]

  row = 0
  @inbounds for i in 1:m
    li, ui = lcon[i], ucon[i]
    (isfinite(li) || isfinite(ui)) || continue
    row += 1
    constraint_rows[i] = row
    push!(rhs, isfinite(li) ? li : ui); push!(y0_std, y0_src[i])
    if li == ui
      con_kind[i] = CON_EQ
    elseif isfinite(li)
      nstd += 1; s = nstd
      con_kind[i] = isfinite(ui) ? CON_RANGE : CON_LB
      con_idx1[i] = s
      push!(I, row); push!(J, s); push!(V, -one(T)); push!(x0_std, zero(T))
      if isfinite(ui)
        nstd += 1; w = nstd
        con_idx2[i] = w
        push!(con_upper_pairs, (i, s, w))
        push!(x0_std, zero(T))
      end
    else
      nstd += 1; s = nstd
      con_kind[i] = CON_UB; con_idx1[i] = s
      push!(I, row); push!(J, s); push!(V, one(T)); push!(x0_std, zero(T))
    end
  end

  # Upper-equality rows (z + w = u - l) for var/con pairs with both bounds finite.
  var_upper_row = zeros(Int, n); con_upper_row = zeros(Int, m)
  for (eq_rows, pairs, l_src, u_src) in
      ((var_upper_row, var_upper_pairs, lvar, uvar),
       (con_upper_row, con_upper_pairs, lcon, ucon))
    for (orig, z, w) in pairs
      row += 1; eq_rows[orig] = row
      push!(rhs, u_src[orig] - l_src[orig]); push!(y0_std, zero(T))
      push!(I, row); push!(J, z); push!(V, one(T))
      push!(I, row); push!(J, w); push!(V, one(T))
    end
  end

  return (
    nstd, nrows = row,
    x0 = x0_std, y0 = y0_std, rhs, x_offset,
    var_lower, var_upper,
    var_start = BoundMap(var_kind, var_idx1, var_idx2, lvar, uvar, Int[]),
    con_start = BoundMap(con_kind, con_idx1, con_idx2, lcon, ucon, constraint_rows),
    var_upper_row, con_upper_row,
    extra_I = I, extra_J = J, extra_V = V,
  )
end

_csc_nz_index(A::AbstractSparseOperator, row::Int, col::Int) =
  _csc_nz_index(operator_sparse_matrix(A), row, col)

function _csc_nz_index(A::SparseMatrixCSC, row::Int, col::Int)
  for p in A.colptr[col]:(A.colptr[col + 1] - 1)
    A.rowval[p] == row && return p
  end
  throw(ArgumentError("missing structural entry ($row, $col) in standard-form matrix"))
end

# Std nzvals = slack/identity rows (kept) + zero where the scatter writes from
# `src`. `_apply_scatter_map!` rebuilds the source-derived entries from `src`
# atop this base each call.
function _scatter_base(stdvals::AbstractVector{T}, dest::AbstractVector{Int}) where {T}
  base = copy(stdvals)
  @inbounds for idx in dest
    base[idx] = zero(T)
  end
  return base
end

# Walk orig Jacobian entries, emit std-form (row, col, scale) into I/J/V *and*
# track (k, scale) for the scatter map. Returns the std `SparseOperator` plus
# the scatter map mapping orig nz_k → std CSC nz position.
function _build_std_A(orig, layout)
  T = eltype(layout.x_offset)
  vs = layout.var_start
  A_rows, A_cols = _sparse_structure(orig.data.A)
  I = copy(layout.extra_I); J = copy(layout.extra_J); V = copy(layout.extra_V)
  src_k = Int[]; src_scale = T[]
  @inbounds for k in eachindex(A_rows)
    row = layout.con_start.row[A_rows[k]]; row == 0 && continue
    _foreach_standard_var(vs, A_cols[k]) do col, s
      push!(I, row); push!(J, col); push!(V, s)
      push!(src_k, k); push!(src_scale, s)
    end
  end
  Astd_csc = sparse(I, J, V, layout.nrows, layout.nstd)
  Astd_op  = sparse_operator(Astd_csc)
  nbase = length(layout.extra_I)
  dest = [_csc_nz_index(Astd_csc, I[nbase + i], J[nbase + i]) for i in eachindex(src_k)]
  return Astd_op, ScatterMap(_scatter_base(_sparse_values(Astd_op), dest), dest, src_k, src_scale)
end

function _build_std_Q(qp::QuadraticModel, layout)
  T = eltype(layout.x_offset)
  vs = layout.var_start
  Q_rows, Q_cols = _sparse_structure(qp.data.Q)
  QI = Int[]; QJ = Int[]; QV = T[]
  src_k = Int[]; src_scale = T[]
  @inbounds for k in eachindex(Q_rows)
    _foreach_standard_var(vs, Q_rows[k]) do a, sa
      _foreach_standard_var(vs, Q_cols[k]) do b, sb
        push!(QI, max(a, b)); push!(QJ, min(a, b)); push!(QV, sa * sb)
        push!(src_k, k); push!(src_scale, sa * sb)
      end
    end
  end
  Qstd_csc = sparse(QI, QJ, QV, layout.nstd, layout.nstd)
  Qstd_op  = sparse_operator(Qstd_csc; symmetric = true)
  dest = [_csc_nz_index(Qstd_csc, QI[i], QJ[i]) for i in eachindex(src_k)]
  return Qstd_op, ScatterMap(_scatter_base(_sparse_values(Qstd_op), dest), dest, src_k, src_scale)
end

function _build_c_map(layout)
  T = eltype(layout.x_offset)
  vs = layout.var_start
  dest = Int[]; src = Int[]; scale = T[]
  @inbounds for i in eachindex(vs.kind)
    _foreach_standard_var(vs, i) do col, s
      push!(dest, col); push!(src, i); push!(scale, s)
    end
  end
  return ScatterMap(zeros(T, layout.nstd), dest, src, scale)
end

@inline function _std_bound_kwargs(layout)
  T = eltype(layout.rhs)
  (lcon = layout.rhs, ucon = copy(layout.rhs),
   lvar = zeros(T, layout.nstd), uvar = fill(T(Inf), layout.nstd))
end

"""
    std, ws = standard_form(orig)

Reformulate `orig` (a [`LinearModel`](@ref), [`QuadraticModel`](@ref), or
[`BatchQuadraticModel`](@ref)) into standard form. Returns the std-form model
and a [`StandardFormWorkspace`](@ref) caching the mapping for incremental
updates via [`update_standard_form!`](@ref).
"""
function standard_form(orig::_ScalarModel)
  T = eltype(orig.data.c)
  layout = _build_standard_layout(orig)
  layout.nstd > 0 || throw(ArgumentError(
    "Standard-form reformulation eliminated all decision variables; trivial all-fixed models are not supported."))
  Astd_op, A_map = _build_std_A(orig, layout)
  has_q = orig isa QuadraticModel
  Qstd_op, Q_map = has_q ? _build_std_Q(orig, layout) :
                            (nothing, ScatterMap(T[], Int[], Int[], T[]))
  c = zeros(T, layout.nstd)
  meta = (x0 = layout.x0, y0 = layout.y0, minimize = orig.meta.minimize, name = orig.meta.name)
  std = if has_q
    QuadraticModel(QPData(Astd_op, c, Qstd_op;
      _std_bound_kwargs(layout)..., c0 = zero(T), _v = zeros(T, layout.nstd)); meta...)
  else
    LinearModel(LPData(Astd_op, c; _std_bound_kwargs(layout)..., c0 = zero(T)); meta...)
  end
  n = NLPModels.get_nvar(orig); m = NLPModels.get_ncon(orig)
  ws = StandardFormWorkspace(
    orig.data.A, has_q ? orig.data.Q : nothing,
    A_map, Q_map, _build_c_map(layout),
    layout.var_start, layout.con_start,
    layout.var_lower, layout.var_upper,
    layout.var_upper_row, layout.con_upper_row,
    copy(layout.rhs), layout.x_offset,
    Vector{T}(undef, m), Vector{T}(undef, m),
    Vector{T}(undef, has_q ? n : 0), Vector{T}(undef, has_q ? n : 0),
    T[], T[],   # c0_batch, c0_tmp — scalar c0 lives on std.data.c0[]
  )
  update_standard_form!(orig, std, ws)
  return std, ws
end

# Pull a single representative instance out of a batched matrix entry. The
# `tag` arg dispatches `_build_scalar_sparse` on backend (CPU vs CUDA ext).
_representative_matrix(op::AbstractSparseOperator, _tag, m, n) = op
_representative_matrix(op::BatchSparseOperator, tag::AbstractVector, m, n) =
  _build_scalar_sparse(tag, op.rows, op.cols, view(op.nzvals, :, 1), m, n)

@inline _build_scalar_sparse(::AbstractVector, rows, cols, vals::AbstractVector, m, n) =
  sparse(rows, cols, vals, m, n)

# Default: no rebuild needed. CUDA ext specializes for `MT <: CuMatrix` to
# rebuild sparse operators with `spmm_ncols = nbatch`.
_adapt_to_batch_backend(qp, ::Type, _nbatch) = qp

function standard_form(bqp::BatchQuadraticModel{T, MT, AOp, QOp}) where {T, MT, AOp, QOp}
  nbatch = bqp.meta.nbatch
  has_q  = bqp.meta.nnzh > 0
  n = NLPModels.get_nvar(bqp); m = NLPModels.get_ncon(bqp)

  # Solve the structural problem once via a representative scalar QP, then
  # broaden the resulting workspace to per-instance shape.
  c_col = bqp.c_batch[:, 1]
  rep = QuadraticModel(QPData(
    _representative_matrix(bqp.A, c_col, m, n), c_col,
    _representative_matrix(bqp.Q, c_col, n, n);
    lvar = bqp.meta.lvar[:, 1], uvar = bqp.meta.uvar[:, 1],
    lcon = bqp.meta.lcon[:, 1], ucon = bqp.meta.ucon[:, 1], c0 = zero(T));
    x0 = bqp.meta.x0[:, 1], y0 = bqp.meta.y0[:, 1],
    minimize = bqp.meta.minimize, name = string(bqp.meta.name, "_rep"))
  std_single, ws_single = standard_form(rep)

  std_batch = BatchQuadraticModel(_adapt_to_batch_backend(std_single, MT, nbatch), nbatch;
    MT, name = std_single.meta.name,
    shared_A = AOp <: AbstractSparseOperator, shared_Q = QOp <: AbstractSparseOperator)

  bm(scalar, l, u) = BoundMap{T, MT, typeof(scalar.idx1), typeof(scalar.kind)}(
    scalar.kind, scalar.idx1, scalar.idx2, copy(l), copy(u), scalar.row)
  WVT = typeof(similar(bqp.meta.lvar, T, 0))
  mat(r, c) = MT(undef, r, c)
  nrows = length(ws_single.rhs_base)

  ws = StandardFormWorkspace(
    bqp.A, has_q ? bqp.Q : nothing,
    ws_single.A_map, ws_single.Q_map, ws_single.c_map,
    bm(ws_single.var_start, bqp.meta.lvar, bqp.meta.uvar),
    bm(ws_single.con_start, bqp.meta.lcon, bqp.meta.ucon),
    ws_single.var_lower, ws_single.var_upper,
    ws_single.var_upper_row, ws_single.con_upper_row,
    mat(nrows, nbatch), mat(n, nbatch),
    mat(m, nbatch), mat(m, nbatch),
    mat(has_q ? n : 0, nbatch), mat(has_q ? n : 0, nbatch),
    WVT(undef, nbatch), WVT(undef, nbatch),
  )
  update_standard_form!(bqp, std_batch, ws)
  return std_batch, ws
end

_A(m::_ScalarModel)        = m.data.A
_Q(m::_ScalarModel)        = m.data.Q
_c(m::_ScalarModel)        = m.data.c
_lcon(m::_ScalarModel)     = m.data.lcon
_ucon(m::_ScalarModel)     = m.data.ucon
_A(m::BatchQuadraticModel) = m.A
_Q(m::BatchQuadraticModel) = m.Q
_c(m::BatchQuadraticModel) = m.c_batch
_lcon(m::BatchQuadraticModel) = m.meta.lcon
_ucon(m::BatchQuadraticModel) = m.meta.ucon

function _scatter_c!(std, ws, src)
  if isempty(ws.qx)
    _apply_scatter_map!(_c(std), ws.c_map, _c(src))
  else
    ws.ctmp .= _c(src) .+ ws.qx
    _apply_scatter_map!(_c(std), ws.c_map, ws.ctmp)
  end
end

# `c0 += scale * dot(a, b)` accumulator. CPU uses `dot`; CUDA ext overrides
# with a non-syncing GEMV so `_set_c0!` stays device-resident.
@inline _add_dot!(c0::AbstractVector{T}, a::AbstractVector{T}, b::AbstractVector{T}, scale::T) where {T} =
  (@inbounds c0[1] += scale * dot(a, b); c0)

function _set_c0!(std::_ScalarModel, ws, src)
  T = eltype(ws.x_offset)
  copyto!(std.data.c0, src.data.c0)
  _add_dot!(std.data.c0, src.data.c, ws.x_offset, one(T))
  isempty(ws.qx) || _add_dot!(std.data.c0, ws.qx, ws.x_offset, T(0.5))
  return std
end

function _set_c0!(::BatchQuadraticModel, ws, src)
  _coldot!(ws.c0_batch, src.c_batch, ws.x_offset)
  ws.c0_batch .+= src.c0_batch
  if !isempty(ws.qx)
    _coldot!(ws.c0_tmp, ws.qx, ws.x_offset)
    ws.c0_batch .+= ws.c0_tmp ./ 2
  end
end

const _ALL_DIRTY = (c=true, c0=true, A=true, Q=true,
                    var_bounds=true, con_bounds=true, x0=true, y0=true)

function _apply!(std, ws::StandardFormWorkspace, src, d)
  has_q = ws.Q_ref !== nothing
  rhs_base_dirty = d.var_bounds || d.con_bounds
  shift_dirty    = d.A          || d.var_bounds
  rhs_dirty      = rhs_base_dirty || shift_dirty
  qx_dirty       = has_q && (d.Q || d.var_bounds)
  c_dirty        = d.c || qx_dirty

  if d.var_bounds
    copyto!(ws.var_start.l, src.meta.lvar)
    copyto!(ws.var_start.u, src.meta.uvar)
    _update_x_offset!(ws.x_offset, ws.var_start)
  end
  if d.con_bounds
    copyto!(ws.con_start.l, src.meta.lcon)
    copyto!(ws.con_start.u, src.meta.ucon)
  end

  d.A && _apply_scatter_map!(_sparse_values(_A(std)), ws.A_map, _sparse_values(ws.A_ref))
  has_q && d.Q && _apply_scatter_map!(_sparse_values(_Q(std)), ws.Q_map, _sparse_values(ws.Q_ref))
  qx_dirty && mul!(ws.qx, ws.Q_ref, ws.x_offset)

  rhs_base_dirty && _update_rhs_base!(ws.rhs_base, ws.var_start, ws.con_start,
                                       ws.var_upper_row, ws.con_upper_row)
  shift_dirty    && mul!(ws.shift, ws.A_ref, ws.x_offset)
  if rhs_dirty
    _apply_rhs_shift!(_lcon(std), ws.rhs_base, ws.shift, ws.con_start.row)
    copyto!(_ucon(std), _lcon(std))
  end

  (d.var_bounds || d.x0) && _update_var_start!(std.meta.x0, src.meta.x0, ws.var_start)
  if d.A || d.x0 || d.con_bounds
    (d.A || d.x0) && mul!(ws.activity, ws.A_ref, src.meta.x0)
    _update_constraint_start!(std.meta.x0, ws.activity, ws.con_start)
  end
  (d.y0 || d.con_bounds) && _update_dual_start!(std.meta.y0, src.meta.y0, ws.con_start.row)

  c_dirty && _scatter_c!(std, ws, src)
  (d.c0 || c_dirty || d.var_bounds) && _set_c0!(std, ws, src)
  return std
end

_absorb_nzvals!(op::AbstractSparseOperator, src) = copyto!(_sparse_values(op), _sparse_values(src))
_absorb_nzvals!(op::BatchSparseOperator, src)    = copyto!(op.nzvals, src)

_set_c0_orig!(orig::_ScalarModel, c0)        = (orig.data.c0 .= c0; orig.data.c0)
_set_c0_orig!(orig::BatchQuadraticModel, c0) = (orig.c0_batch .= c0; orig.c0_batch)

"""
    update_standard_form!(orig, std, ws; c, c0, A, Q, lvar, uvar, lcon, ucon, x0, y0)

Absorb any provided fields into `orig`, then propagate the minimal set of
updates to `std`. With no kwargs, performs a full refresh from `orig`. Caller
must not change sparsity patterns or bound kinds (finite ↔ infinite,
`l == u`) — rebuild with [`standard_form`](@ref) for structural changes.
"""
function update_standard_form!(orig, std, ws::StandardFormWorkspace;
                               c = nothing, c0 = nothing, A = nothing, Q = nothing,
                               lvar = nothing, uvar = nothing, lcon = nothing, ucon = nothing,
                               x0 = nothing, y0 = nothing)
  c    === nothing || copyto!(_c(orig), c)
  c0   === nothing || _set_c0_orig!(orig, c0)
  A    === nothing || _absorb_nzvals!(_A(orig), A)
  Q    === nothing || _absorb_nzvals!(_Q(orig), Q)
  lvar === nothing || copyto!(orig.meta.lvar, lvar)
  uvar === nothing || copyto!(orig.meta.uvar, uvar)
  lcon === nothing || copyto!(orig.meta.lcon, lcon)
  ucon === nothing || copyto!(orig.meta.ucon, ucon)
  x0   === nothing || copyto!(orig.meta.x0, x0)
  y0   === nothing || copyto!(orig.meta.y0, y0)
  any_set = !all(isnothing, (c, c0, A, Q, lvar, uvar, lcon, ucon, x0, y0))
  d = any_set ? (
    c = c !== nothing, c0 = c0 !== nothing,
    A = A !== nothing, Q = Q !== nothing,
    var_bounds = lvar !== nothing || uvar !== nothing,
    con_bounds = lcon !== nothing || ucon !== nothing,
    x0 = x0 !== nothing, y0 = y0 !== nothing,
  ) : _ALL_DIRTY
  _apply!(std, ws, orig, d)
  return std
end

"""
    recover_primal!(x, ws, z)

Undo the std-form reformulation: write the original primal `x` from the
std-form solution `z`. Vectors for scalar workspaces, matrices for batch.
"""
function recover_primal!(x::AbstractVecOrMat, ws::StandardFormWorkspace, z::AbstractVecOrMat)
  copyto!(x, ws.x_offset)
  _recover_primal_apply!(x, ws.var_start.kind, ws.var_start.idx1, ws.var_start.idx2, z)
  return x
end

recover_primal(ws::StandardFormWorkspace, z::AbstractVecOrMat) =
  recover_primal!(similar(ws.x_offset), ws, z)

"""
    recover_variable_multipliers!(zl, zu, ws, zstd)

Scatter std-form non-negativity multipliers `zstd` back to original lower/upper
bound multipliers (`zl`, `zu`).
"""
function recover_variable_multipliers!(zl::AbstractVecOrMat, zu::AbstractVecOrMat,
                                       ws::StandardFormWorkspace, zstd::AbstractVecOrMat)
  _scatter_multipliers!(zl, zu, ws.var_lower, ws.var_upper, zstd)
  return zl, zu
end
