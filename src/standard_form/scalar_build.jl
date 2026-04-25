# ---------- layout construction (CPU) ----------
# The layout build is intrinsically scalar-sequential (each iteration depends
# on the running `nstd`/`row`/`nz` counters), so we host-pull GPU bound vectors
# once at the top and drive the rest with plain `Vector`s on CPU.

# A constraint is kept in std form iff at least one side is finite (one-sided
# inequalities stay; fully free rows are dropped).
@inline _has_standard_row(l, u) = isfinite(l) || isfinite(u)

# No-op on CPU; pulls device vectors to host so the scalar-indexed layout
# loops can run without `@allowscalar`.
_host_vector(v::Vector) = v
_host_vector(v::AbstractVector) = collect(v)

function _count_standard_variables(lvar, uvar)
  nstd = nupper = 0
  @inbounds for i in eachindex(lvar, uvar)
    li = lvar[i]; ui = uvar[i]
    li == ui && continue
    if isfinite(li)
      nstd += 1
      isfinite(ui) && (nstd += 1; nupper += 1)
    elseif isfinite(ui)
      nstd += 1
    else
      nstd += 2
    end
  end
  return (; nstd, nupper)
end

function _count_standard_constraints(lcon, ucon)
  nprimary = nstd = nupper = nslack_nnz = 0
  @inbounds for i in eachindex(lcon, ucon)
    li = lcon[i]; ui = ucon[i]
    _has_standard_row(li, ui) || continue
    nprimary += 1
    if li != ui
      nstd += 1; nslack_nnz += 1
      if isfinite(li) && isfinite(ui)
        nstd += 1; nupper += 1
      end
    end
  end
  return (; nprimary, nstd, nupper, nslack_nnz)
end

# Primary constraint rhs: prefer the lower bound; fall back to upper when l = -Inf.
_primary_rhs(l, u) = isfinite(l) ? l : u

# Per-side upper-bound bundle for `_build_*_layout!` / `_emit_upper_rows!`.
# `n_upper` = how many orig vars/cons have both finite l and u; `n_orig` = total
# orig vars/cons (sized so `eq_rows[orig]` indexes by original index).
@inline _upper_bundle(n_upper::Int, n_orig::Int) =
  (orig_idx = Vector{Int}(undef, n_upper),
   zidx     = Vector{Int}(undef, n_upper),
   widx     = Vector{Int}(undef, n_upper),
   eq_rows  = zeros(Int, n_orig))

function _build_variable_layout!(
  x_offset::Vector{T}, var_lower::Vector{Int}, var_upper::Vector{Int},
  var_kind::Vector{BoundKind}, var_idx1::Vector{Int}, var_idx2::Vector{Int},
  var_upper_bundle, x0_std::Vector{T}, x0_src, lvar, uvar,
) where {T}
  nstd = 0
  kupper = 1
  @inbounds for i in eachindex(lvar, uvar)
    li = lvar[i]; ui = uvar[i]
    if li == ui
      x_offset[i] = li
    elseif isfinite(li)
      nstd += 1
      x_offset[i] = li
      var_lower[i] = nstd
      var_idx1[i] = nstd
      x0_std[nstd] = x0_src[i] - li
      if isfinite(ui)
        nstd += 1
        var_kind[i] = VAR_LB_UB
        var_upper[i] = nstd
        var_idx2[i] = nstd
        x0_std[nstd] = ui - x0_src[i]
        var_upper_bundle.orig_idx[kupper] = i
        var_upper_bundle.zidx[kupper] = var_lower[i]
        var_upper_bundle.widx[kupper] = var_upper[i]
        kupper += 1
      else
        var_kind[i] = VAR_LB
      end
    elseif isfinite(ui)
      nstd += 1
      x_offset[i] = ui
      var_kind[i] = VAR_UB
      var_upper[i] = nstd
      var_idx1[i] = nstd
      x0_std[nstd] = ui - x0_src[i]
    else
      nstd += 1
      var_kind[i] = VAR_FREE
      var_idx1[i] = nstd
      xi = x0_src[i]
      x0_std[nstd] = max(xi, zero(T))
      nstd += 1
      var_idx2[i] = nstd
      x0_std[nstd] = max(-xi, zero(T))
    end
  end
  return nstd
end

function _build_constraint_layout!(
  constraint_rows::Vector{Int},
  con_kind::Vector{BoundKind}, con_idx1::Vector{Int}, con_idx2::Vector{Int},
  con_upper_bundle,
  rhs::Vector{T}, y0_std::Vector{T},
  I::Vector{Int}, J::Vector{Int}, V::Vector{T},
  y0_src, lcon, ucon, nstd::Int,
) where {T}
  row = 0; nz = 1; kupper = 1
  @inbounds for i in eachindex(lcon, ucon)
    li = lcon[i]; ui = ucon[i]
    _has_standard_row(li, ui) || continue
    row += 1
    constraint_rows[i] = row
    rhs[row] = _primary_rhs(li, ui)
    y0_std[row] = y0_src[i]
    if li == ui
      con_kind[i] = CON_EQ
    elseif isfinite(li)
      nstd += 1; s = nstd
      con_kind[i] = isfinite(ui) ? CON_RANGE : CON_LB
      con_idx1[i] = s
      I[nz] = row; J[nz] = s; V[nz] = -one(T); nz += 1
      if isfinite(ui)
        nstd += 1; w = nstd
        con_idx2[i] = w
        con_upper_bundle.orig_idx[kupper] = i
        con_upper_bundle.zidx[kupper] = s
        con_upper_bundle.widx[kupper] = w
        kupper += 1
      end
    else
      nstd += 1; s = nstd
      con_kind[i] = CON_UB
      con_idx1[i] = s
      I[nz] = row; J[nz] = s; V[nz] = one(T); nz += 1
    end
  end
  return row, nstd, nz
end

# For each (orig variable / orig constraint) that has both a finite lower and
# upper bound, emit one upper-equality row `z + w = u - l` (where z and w are
# the std-form lower and upper slack indices for that orig). `upper` is a
# NamedTuple `(orig_idx, zidx, widx, eq_rows)` packing the upper-bound metadata.
# `y0_std` is pre-zeroed; upper-row entries keep that zero.
function _emit_upper_rows!(rhs, I, J, V::Vector{T}, row, nz, upper, l, u) where {T}
  @inbounds for k in eachindex(upper.orig_idx)
    orig = upper.orig_idx[k]
    row += 1
    upper.eq_rows[orig] = row
    rhs[row] = u[orig] - l[orig]
    I[nz] = row; J[nz] = upper.zidx[k]; V[nz] = one(T); nz += 1
    I[nz] = row; J[nz] = upper.widx[k]; V[nz] = one(T); nz += 1
  end
  return row, nz
end

function _build_standard_layout(model)
  T = eltype(model.data.c)
  n = NLPModels.get_nvar(model); m = NLPModels.get_ncon(model)
  lvar = _host_vector(model.meta.lvar); uvar = _host_vector(model.meta.uvar)
  lcon = _host_vector(model.meta.lcon); ucon = _host_vector(model.meta.ucon)
  x0_src = _host_vector(model.meta.x0); y0_src = _host_vector(model.meta.y0)

  vc    = _count_standard_variables(lvar, uvar)
  cc    = _count_standard_constraints(lcon, ucon)
  nstd  = vc.nstd + cc.nstd
  nrows = cc.nprimary + vc.nupper + cc.nupper

  x_offset = zeros(T, n)
  var_lower = zeros(Int, n); var_upper = zeros(Int, n)
  var_kind = fill(BK_NONE, n)
  var_idx1 = zeros(Int, n); var_idx2 = zeros(Int, n)
  var_upper_bundle = _upper_bundle(vc.nupper, n)
  x0_std = zeros(T, nstd)

  _build_variable_layout!(x_offset, var_lower, var_upper, var_kind, var_idx1, var_idx2,
    var_upper_bundle, x0_std, x0_src, lvar, uvar)

  constraint_rows = zeros(Int, m)
  con_kind = fill(BK_NONE, m)
  con_idx1 = zeros(Int, m); con_idx2 = zeros(Int, m)
  con_upper_bundle = _upper_bundle(cc.nupper, m)
  rhs = zeros(T, nrows); y0_std = zeros(T, nrows)

  nnz_total = cc.nslack_nnz + 2vc.nupper + 2cc.nupper
  I = Vector{Int}(undef, nnz_total); J = Vector{Int}(undef, nnz_total); V = Vector{T}(undef, nnz_total)

  row, filled_nstd, nz = _build_constraint_layout!(constraint_rows, con_kind, con_idx1, con_idx2,
    con_upper_bundle, rhs, y0_std, I, J, V, y0_src, lcon, ucon, vc.nstd)
  filled_nstd == nstd || throw(ArgumentError("standard-form variable count mismatch"))
  row, nz = _emit_upper_rows!(rhs, I, J, V, row, nz, var_upper_bundle, lvar, uvar)
  row, nz = _emit_upper_rows!(rhs, I, J, V, row, nz, con_upper_bundle, lcon, ucon)
  row == nrows || throw(ArgumentError("standard-form row count mismatch"))

  var_start = BoundMap(var_kind, var_idx1, var_idx2, lvar, uvar, Int[])
  con_start = BoundMap(con_kind, con_idx1, con_idx2, lcon, ucon, constraint_rows)
  return StandardFormLayout(nstd, nrows, x0_std, y0_std, rhs, x_offset,
    var_lower, var_upper, var_start, con_start,
    var_upper_bundle.eq_rows, con_upper_bundle.eq_rows, I, J, V)
end

# ---------- scatter maps, workspace construction ----------

@inline function _foreach_standard_var(f, vs::BoundMap, idx::Int)
  kind = vs.kind[idx]
  if kind == VAR_LB || kind == VAR_LB_UB
    f(vs.idx1[idx], one(eltype(vs.l)))
  elseif kind == VAR_UB
    f(vs.idx1[idx], -one(eltype(vs.l)))
  elseif kind == VAR_FREE
    f(vs.idx1[idx], one(eltype(vs.l)))
    f(vs.idx2[idx], -one(eltype(vs.l)))
  end
  return nothing
end

_csc_nz_index(A::AbstractSparseOperator, row::Int, col::Int) =
  _csc_nz_index(operator_sparse_matrix(A), row, col)

function _csc_nz_index(A::SparseMatrixCSC, row::Int, col::Int)
  for p in A.colptr[col]:(A.colptr[col + 1] - 1)
    A.rowval[p] == row && return p
  end
  throw(ArgumentError("missing structural entry ($row, $col) in standard-form matrix"))
end

function _count_standard_jacobian_entries(A_rows, A_cols, row_map, var_kind)
  nnz = 0
  @inbounds for k in eachindex(A_rows, A_cols)
    row_map[A_rows[k]] == 0 && continue
    nnz += _standard_var_width(var_kind[A_cols[k]])
  end
  return nnz
end

function _fill_standard_jacobian!(I, J, V, nz, row_map, vs::BoundMap, A_rows, A_cols)
  @inbounds for k in eachindex(A_rows, A_cols)
    row = row_map[A_rows[k]]
    row == 0 && continue
    _foreach_standard_var(vs, A_cols[k]) do col, scale
      I[nz] = row; J[nz] = col; V[nz] = scale
      nz += 1
    end
  end
  return nz
end

function _count_standard_hessian_entries(Q_rows, Q_cols, vs::BoundMap)
  nnz = 0
  @inbounds for k in eachindex(Q_rows, Q_cols)
    nnz += _standard_var_width(vs.kind[Q_rows[k]]) * _standard_var_width(vs.kind[Q_cols[k]])
  end
  return nnz
end

function _fill_standard_hessian!(QI, QJ, QV, vs::BoundMap, Q_rows, Q_cols)
  kq = 1
  @inbounds for k in eachindex(Q_rows, Q_cols)
    _foreach_standard_var(vs, Q_rows[k]) do a, scale_a
      _foreach_standard_var(vs, Q_cols[k]) do b, scale_b
        QI[kq] = max(a, b); QJ[kq] = min(a, b); QV[kq] = scale_a * scale_b
        kq += 1
      end
    end
  end
  return kq
end

# Build the std-form Jacobian and the bound vectors shared by LP and QP.
function _build_std_jacobian(layout::StandardFormLayout{T}, A_rows, A_cols) where {T}
  vs = layout.var_start
  nnz_a = _count_standard_jacobian_entries(A_rows, A_cols, layout.con_start.row, vs.kind)
  nbase = length(layout.extra_I)
  ntotal = nbase + nnz_a
  I = Vector{Int}(undef, ntotal); J = Vector{Int}(undef, ntotal); V = Vector{T}(undef, ntotal)
  copyto!(I, 1, layout.extra_I, 1, nbase)
  copyto!(J, 1, layout.extra_J, 1, nbase)
  copyto!(V, 1, layout.extra_V, 1, nbase)
  nz = _fill_standard_jacobian!(I, J, V, nbase + 1, layout.con_start.row, vs, A_rows, A_cols)
  nz == ntotal + 1 || throw(ArgumentError("standard-form Jacobian count mismatch"))
  return sparse(I, J, V, layout.nrows, layout.nstd)
end

@inline _std_bound_kwargs(layout::StandardFormLayout{T}) where {T} =
  (lcon = layout.rhs, ucon = copy(layout.rhs),
   lvar = zeros(T, layout.nstd), uvar = fill(T(Inf), layout.nstd))

function _build_standard_linear_data(layout::StandardFormLayout{T}, A_rows, A_cols) where {T}
  Astd = _build_std_jacobian(layout, A_rows, A_cols)
  return LPData(Astd, zeros(T, layout.nstd); _std_bound_kwargs(layout)..., c0 = zero(T))
end

function _build_standard_quadratic_data(layout::StandardFormLayout{T}, A_rows, A_cols, Q_rows, Q_cols) where {T}
  Astd = _build_std_jacobian(layout, A_rows, A_cols)
  vs = layout.var_start
  nnz_q = _count_standard_hessian_entries(Q_rows, Q_cols, vs)
  QI = Vector{Int}(undef, nnz_q); QJ = Vector{Int}(undef, nnz_q); QV = Vector{T}(undef, nnz_q)
  _fill_standard_hessian!(QI, QJ, QV, vs, Q_rows, Q_cols) == nnz_q + 1 || throw(ArgumentError("standard-form Hessian count mismatch"))
  Qstd = sparse(QI, QJ, QV, layout.nstd, layout.nstd)
  return QPData(Astd, zeros(T, layout.nstd), Qstd;
    _std_bound_kwargs(layout)..., c0 = zero(T), _v = zeros(T, layout.nstd))
end

# Std-form `c` is `c_map.scale[k] * src.c[c_map.src[k]]` summed into `c_map.dest[k]`,
# atop a zero base. One destination entry per standard-var width unit.
function _build_c_map(layout::StandardFormLayout{T}) where {T}
  vs = layout.var_start
  nc = sum(_standard_var_width, vs.kind; init = 0)
  dest = Vector{Int}(undef, nc); src = Vector{Int}(undef, nc); scale = Vector{T}(undef, nc)
  k = 1
  @inbounds for i in eachindex(vs.kind)
    _foreach_standard_var(vs, i) do col, s
      dest[k] = col; src[k] = i; scale[k] = s
      k += 1
    end
  end
  return ScatterMap(zeros(T, layout.nstd), dest, src, scale)
end

# Std-form `A` adds `A_map.scale[k] * src.A.nzvals[A_map.src[k]]` into the std
# CSC nzvals slot `A_map.dest[k]`. The slack/identity-row nonzeros stay as the
# `A_base` snapshot (they don't depend on src.A values).
function _build_A_map(orig, Astd, layout::StandardFormLayout{T}) where {T}
  A_rows, A_cols = _sparse_structure(orig.data.A)
  vs = layout.var_start
  na = _count_standard_jacobian_entries(A_rows, A_cols, layout.con_start.row, vs.kind)
  dest = Vector{Int}(undef, na); src = Vector{Int}(undef, na); scale = Vector{T}(undef, na)
  ka = 1
  @inbounds for k in eachindex(A_rows)
    row = layout.con_start.row[A_rows[k]]
    row == 0 && continue
    _foreach_standard_var(vs, A_cols[k]) do col, s
      dest[ka] = _csc_nz_index(Astd, row, col)
      src[ka] = k; scale[ka] = s
      ka += 1
    end
  end
  base = copy(_sparse_values(Astd))
  @inbounds for idx in dest
    base[idx] = zero(T)
  end
  return ScatterMap(base, dest, src, scale)
end

# Std-form Hessian scatter: each orig (i, j) entry expands per-row/col bound
# kind into one dest slot per (a, b) std var pair.
function _build_Q_map(qp::QuadraticModel, Qstd, layout::StandardFormLayout{T}) where {T}
  Q_rows, Q_cols = _sparse_structure(qp.data.Q)
  vs = layout.var_start
  nq = _count_standard_hessian_entries(Q_rows, Q_cols, vs)
  dest = Vector{Int}(undef, nq); src_idx = Vector{Int}(undef, nq); scale = Vector{T}(undef, nq)
  kq = 1
  @inbounds for k in eachindex(Q_rows)
    _foreach_standard_var(vs, Q_rows[k]) do a, sa
      _foreach_standard_var(vs, Q_cols[k]) do b, sb
        dest[kq] = _csc_nz_index(Qstd, max(a, b), min(a, b))
        src_idx[kq] = k; scale[kq] = sa * sb
        kq += 1
      end
    end
  end
  base = copy(_sparse_values(Qstd))
  @inbounds for idx in dest
    base[idx] = zero(T)
  end
  return ScatterMap(base, dest, src_idx, scale)
end

# QP-side workspace fields (`Q_ref`, `Q_src`, `Q_map`, `qx`, `ctmp`). LP returns
# zero-length placeholders so the StandardFormWorkspace layout stays uniform;
# `_apply!` gates the QP branch on `ws.Q_ref === nothing`.
_qp_scratch(::LinearModel, _std_data, ::StandardFormLayout{T}) where {T} =
  (nothing, T[], ScatterMap(T[], Int[], Int[], T[]), T[], T[])

function _qp_scratch(qp::QuadraticModel, std_data, layout::StandardFormLayout{T}) where {T}
  n = NLPModels.get_nvar(qp)
  return (qp.data.Q,
          Vector{T}(undef, _nnz(qp.data.Q)),
          _build_Q_map(qp, std_data.Q, layout),
          Vector{T}(undef, n),
          Vector{T}(undef, n))
end

function _build_standard_workspace(orig::_ScalarModel, std_data, layout::StandardFormLayout{T}) where {T}
  A_map = _build_A_map(orig, std_data.A, layout)
  c_map = _build_c_map(layout)
  m = NLPModels.get_ncon(orig)
  Q_ref, Q_src, Q_map, qx, ctmp = _qp_scratch(orig, std_data, layout)
  return StandardFormWorkspace(
    orig.data.A, similar(_sparse_values(orig.data.A)), A_map, c_map,
    _structure_signature(orig),
    copy(layout.rhs), layout.x_offset,
    layout.var_start, layout.con_start,
    layout.var_lower, layout.var_upper,
    layout.var_upper_row, layout.con_upper_row,
    Vector{T}(undef, m), Vector{T}(undef, m),
    Q_ref, Q_src, Q_map, qx, ctmp,
  )
end

