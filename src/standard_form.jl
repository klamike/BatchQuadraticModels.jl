# Standard-form reformulation of LinearModel/QuadraticModel into the canonical
#
#   min  c' z + c0        s.t.  A z = b,  z >= 0
#
# with (optionally) a quadratic term (1/2) z' Q z. The transform splits each
# original variable into one or two standard variables (lower/upper/ranged/free)
# and adds slack variables for each inequality constraint. A `StandardFormWorkspace`
# caches the mapping so that value updates on the original problem can be pushed
# through to the standard form incrementally via `update_standard_form!`.
#
# Bound kinds:
#   VAR_LB       l finite, u infinite      -> z = x - l, x = l + z
#   VAR_LB_UB    l and u finite, l != u    -> z = x - l, w = u - x
#   VAR_UB       l infinite, u finite      -> z = u - x
#   VAR_FREE     both infinite             -> x = z1 - z2
#   VAR_EQ       l == u (fixed)            -> no standard-form variable
#   CON_EQ       l == u                    -> row kept, no slack
#   CON_LB       l finite, u infinite      -> row + slack, Ax - s = l
#   CON_RANGE    both finite               -> row + slack + upper-complement
#   CON_UB       l infinite, u finite      -> row + slack, Ax + s = u
@enum BoundKind::UInt8 BK_NONE VAR_LB VAR_LB_UB VAR_UB VAR_FREE VAR_EQ CON_EQ CON_LB CON_RANGE CON_UB

struct ScatterMap{T, VT <: AbstractVector{T}, VI <: AbstractVector{Int}}
  base::VT
  dest::VI
  src::VI
  scale::VT
end

# `row` is populated for constraint maps (gives standard-form row for each source
# constraint) and left empty for variable maps.
struct BoundMap{T, VT <: AbstractVector{T}, VI <: AbstractVector{Int}, VU <: AbstractVector{BoundKind}}
  kind::VU
  idx1::VI
  idx2::VI
  l::VT
  u::VT
  row::VI
end

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
  # Hessian fields: MQ = Nothing for LP, otherwise the original symmetric operator.
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

# ---------- solution recovery ----------

"""
    recover_primal!(x, ws, z)

Undo the standard-form reformulation: given the standard-form solution `z`,
write the original primal `x` to the preallocated buffer.
"""
function recover_primal!(x::AbstractVector, ws::StandardFormWorkspace, z::AbstractVector)
  copyto!(x, ws.x_offset)
  _recover_primal_apply!(x, ws.var_start.kind, ws.var_start.idx1, ws.var_start.idx2, z)
  return x
end

recover_primal(ws::StandardFormWorkspace, z::AbstractVector) = recover_primal!(similar(ws.x_offset), ws, z)

function _recover_primal_apply!(x::Vector, kind, idx1, idx2, z)
  @inbounds for i in eachindex(x)
    k = kind[i]
    if k == VAR_LB || k == VAR_LB_UB
      x[i] += z[idx1[i]]
    elseif k == VAR_UB
      x[i] -= z[idx1[i]]
    elseif k == VAR_FREE
      x[i] += z[idx1[i]] - z[idx2[i]]
    end
  end
  return x
end

"""
    recover_variable_multipliers!(zl, zu, ws, zstd)

Scatter the standard-form variable multipliers `zstd` into the original
lower/upper bound multipliers `zl`/`zu`.
"""
function recover_variable_multipliers!(zl::AbstractVector, zu::AbstractVector, ws::StandardFormWorkspace, zstd::AbstractVector)
  _scatter_multipliers!(zl, zu, ws.var_lower, ws.var_upper, zstd)
  return zl, zu
end

function _gather_dual!(mult::Vector, rows, y)
  fill!(mult, zero(eltype(mult)))
  @inbounds for i in eachindex(mult)
    row = rows[i]
    row > 0 && (mult[i] = y[row])
  end
  return mult
end

function _scatter_multipliers!(zl::Vector, zu::Vector, var_lower, var_upper, zstd)
  fill!(zl, zero(eltype(zl)))
  fill!(zu, zero(eltype(zu)))
  @inbounds for i in eachindex(zl)
    li, ui = var_lower[i], var_upper[i]
    li > 0 && (zl[i] = zstd[li])
    ui > 0 && (zu[i] = zstd[ui])
  end
  return zl, zu
end

# ---------- structural hashing (reject incompatible updates) ----------

function _source_structure(A::Union{SparseMatrixCSC, SparseMatrixCOO})
  rows = Vector{Int}(undef, _nnz(A))
  cols = similar(rows)
  _copy_sparse_structure!(A, rows, cols)
  return rows, cols
end

_source_structure(A) = _source_structure(operator_sparse_matrix(A))

_structure_hash(A) = _structure_hash(operator_sparse_matrix(A))
_structure_hash(A::SparseMatrixCSC) = hash((size(A), A.colptr, A.rowval))
_structure_hash(A::SparseMatrixCOO) = hash((A.rows, A.cols, size(A)))

function _bound_type_code(l, u)
  lfin = isfinite(l)
  ufin = isfinite(u)
  if lfin
    if ufin
      return l == u ? VAR_EQ : VAR_LB_UB
    end
    return VAR_LB
  elseif ufin
    return VAR_UB
  else
    return VAR_FREE
  end
end

function _bounds_signature(seed::UInt, l::AbstractVector, u::AbstractVector)
  h = seed
  lh, uh = Array(l), Array(u)
  @inbounds for i in eachindex(lh, uh)
    h = hash(_bound_type_code(lh[i], uh[i]), h)
  end
  return h
end

function _structure_signature(model::Union{LinearModel, QuadraticModel})
  data = model.data
  h = hash((size(data.A), NLPModels.get_nvar(model), NLPModels.get_ncon(model)))
  h = hash(_structure_hash(data.A), h)
  if hasproperty(data, :Q)
    h = hash(size(data.Q), h)
    h = hash(_structure_hash(data.Q), h)
  end
  h = _bounds_signature(h, model.meta.lvar, model.meta.uvar)
  h = _bounds_signature(h, model.meta.lcon, model.meta.ucon)
  return h
end

_sparse_nzvals(A::SparseMatrixCSC) = SparseArrays.nonzeros(A)
_sparse_nzvals(A) = _sparse_nzvals(operator_sparse_matrix(A))

@inline function _standard_var_width(kind::BoundKind)
  return kind == VAR_FREE ? 2 : kind == BK_NONE ? 0 : 1
end

function _apply_scatter_map!(dest::AbstractVector{T}, map::ScatterMap{T}, src::AbstractVector{T}) where {T}
  copyto!(dest, map.base)
  @inbounds for k in eachindex(map.dest)
    dest[map.dest[k]] += map.scale[k] * src[map.src[k]]
  end
  return dest
end

# ---------- layout construction (CPU) ----------

@inline _has_standard_row(l, u) = isfinite(l) || isfinite(u)
_host_vector(v::Vector) = v
_host_vector(v::AbstractVector) = collect(v)

function _count_standard_variables(lvar, uvar)
  nstd = 0
  nupper = 0
  @inbounds for i in eachindex(lvar, uvar)
    li = lvar[i]; ui = uvar[i]
    if li == ui
      continue
    elseif isfinite(li)
      nstd += 1
      if isfinite(ui)
        nstd += 1
        nupper += 1
      end
    elseif isfinite(ui)
      nstd += 1
    else
      nstd += 2
    end
  end
  return nstd, nupper
end

function _count_standard_constraints(lcon, ucon)
  nrows = 0; nslack = 0; nupper = 0; nnz = 0
  @inbounds for i in eachindex(lcon, ucon)
    li = lcon[i]; ui = ucon[i]
    _has_standard_row(li, ui) || continue
    nrows += 1
    if li != ui
      nslack += 1
      nnz += 1
      if isfinite(li) && isfinite(ui)
        nslack += 1
        nupper += 1
      end
    end
  end
  return nrows, nslack, nupper, nnz
end

function _primary_rhs(l::T, u::T, shift::T) where {T}
  return isfinite(l) ? l - shift : u - shift
end

function _split_free_start(x::T) where {T}
  return x >= zero(T) ? (x, zero(T)) : (zero(T), -x)
end

function _build_variable_layout!(
  x_offset::Vector{T}, var_lower::Vector{Int}, var_upper::Vector{Int},
  var_kind::Vector{BoundKind}, var_idx1::Vector{Int}, var_idx2::Vector{Int},
  var_upper_orig::Vector{Int}, var_upper_zidx::Vector{Int}, var_upper_widx::Vector{Int},
  x0_std::Vector{T}, x0_src, lvar, uvar,
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
        var_upper_orig[kupper] = i
        var_upper_zidx[kupper] = var_lower[i]
        var_upper_widx[kupper] = var_upper[i]
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
      pos_start, neg_start = _split_free_start(x0_src[i])
      x0_std[nstd] = pos_start
      nstd += 1
      var_idx2[i] = nstd
      x0_std[nstd] = neg_start
    end
  end
  return nstd
end

function _build_constraint_layout!(
  row_map::Vector{Int}, constraint_rows::Vector{Int},
  con_kind::Vector{BoundKind}, con_idx1::Vector{Int}, con_idx2::Vector{Int},
  con_upper_orig::Vector{Int}, con_upper_sidx::Vector{Int}, con_upper_widx::Vector{Int},
  rhs::Vector{T}, y0_std::Vector{T}, x0_std::Vector{T},
  I::Vector{Int}, J::Vector{Int}, V::Vector{T},
  y0_src, lcon, ucon, nstd::Int,
) where {T}
  row = 0; nz = 1; kupper = 1
  @inbounds for i in eachindex(lcon, ucon)
    li = lcon[i]; ui = ucon[i]
    _has_standard_row(li, ui) || continue
    row += 1
    row_map[i] = row
    constraint_rows[i] = row
    rhs[row] = _primary_rhs(li, ui, zero(T))
    y0_std[row] = y0_src[i]
    if li == ui
      con_kind[i] = CON_EQ
    elseif isfinite(li)
      nstd += 1; s = nstd
      con_kind[i] = isfinite(ui) ? CON_RANGE : CON_LB
      con_idx1[i] = s
      I[nz] = row; J[nz] = s; V[nz] = -one(T); nz += 1
      x0_std[s] = zero(T)
      if isfinite(ui)
        nstd += 1; w = nstd
        con_idx2[i] = w
        con_upper_orig[kupper] = i
        con_upper_sidx[kupper] = s
        con_upper_widx[kupper] = w
        kupper += 1
        x0_std[w] = zero(T)
      end
    else
      nstd += 1; s = nstd
      con_kind[i] = CON_UB
      con_idx1[i] = s
      I[nz] = row; J[nz] = s; V[nz] = one(T); nz += 1
      x0_std[s] = zero(T)
    end
  end
  return row, nstd, nz
end

function _emit_upper_rows!(rhs, y0_std, I, J, V, row, nz, eq_rows, orig_idx, zidx, widx, l, u, ::Type{T}) where {T}
  @inbounds for k in eachindex(orig_idx)
    orig = orig_idx[k]
    row += 1
    eq_rows[orig] = row
    rhs[row] = u[orig] - l[orig]
    y0_std[row] = zero(T)
    I[nz] = row; J[nz] = zidx[k]; V[nz] = one(T); nz += 1
    I[nz] = row; J[nz] = widx[k]; V[nz] = one(T); nz += 1
  end
  return row, nz
end

function _fill_upper_rows!(rhs::Vector{T}, y0_std, I, J, V, row, nz,
  var_upper_eq_rows, con_upper_eq_rows,
  var_upper_orig, var_upper_zidx, var_upper_widx,
  con_upper_orig, con_upper_sidx, con_upper_widx,
  lvar, uvar, lcon, ucon) where {T}
  row, nz = _emit_upper_rows!(rhs, y0_std, I, J, V, row, nz, var_upper_eq_rows, var_upper_orig, var_upper_zidx, var_upper_widx, lvar, uvar, T)
  row, nz = _emit_upper_rows!(rhs, y0_std, I, J, V, row, nz, con_upper_eq_rows, con_upper_orig, con_upper_sidx, con_upper_widx, lcon, ucon, T)
  return row, nz
end

function _build_standard_layout(model)
  T = eltype(model.data.c)
  n = NLPModels.get_nvar(model); m = NLPModels.get_ncon(model)
  lvar = _host_vector(model.meta.lvar); uvar = _host_vector(model.meta.uvar)
  lcon = _host_vector(model.meta.lcon); ucon = _host_vector(model.meta.ucon)
  x0_src = _host_vector(model.meta.x0); y0_src = _host_vector(model.meta.y0)

  nstd_var, nvar_upper = _count_standard_variables(lvar, uvar)
  nprimary, nstd_con, ncon_upper, nprimary_slack_nnz = _count_standard_constraints(lcon, ucon)
  nstd = nstd_var + nstd_con
  nrows = nprimary + nvar_upper + ncon_upper

  x_offset = zeros(T, n)
  var_lower = zeros(Int, n); var_upper = zeros(Int, n)
  var_kind = fill(BK_NONE, n)
  var_idx1 = zeros(Int, n); var_idx2 = zeros(Int, n)
  var_upper_eq_rows = zeros(Int, n)
  var_upper_orig = Vector{Int}(undef, nvar_upper)
  var_upper_zidx = Vector{Int}(undef, nvar_upper)
  var_upper_widx = Vector{Int}(undef, nvar_upper)
  x0_std = zeros(T, nstd)

  _build_variable_layout!(x_offset, var_lower, var_upper, var_kind, var_idx1, var_idx2,
    var_upper_orig, var_upper_zidx, var_upper_widx, x0_std, x0_src, lvar, uvar)

  row_map = zeros(Int, m); constraint_rows = zeros(Int, m)
  con_kind = fill(BK_NONE, m)
  con_idx1 = zeros(Int, m); con_idx2 = zeros(Int, m)
  con_upper_eq_rows = zeros(Int, m)
  con_upper_orig = Vector{Int}(undef, ncon_upper)
  con_upper_sidx = Vector{Int}(undef, ncon_upper)
  con_upper_widx = Vector{Int}(undef, ncon_upper)
  rhs = zeros(T, nrows); y0_std = zeros(T, nrows)

  nnz_total = nprimary_slack_nnz + 2nvar_upper + 2ncon_upper
  I = Vector{Int}(undef, nnz_total); J = Vector{Int}(undef, nnz_total); V = Vector{T}(undef, nnz_total)

  row, filled_nstd, nz = _build_constraint_layout!(row_map, constraint_rows, con_kind, con_idx1, con_idx2,
    con_upper_orig, con_upper_sidx, con_upper_widx, rhs, y0_std, x0_std, I, J, V, y0_src, lcon, ucon, nstd_var)
  filled_nstd == nstd || throw(ArgumentError("standard-form variable count mismatch"))
  row, nz = _fill_upper_rows!(rhs, y0_std, I, J, V, row, nz,
    var_upper_eq_rows, con_upper_eq_rows,
    var_upper_orig, var_upper_zidx, var_upper_widx,
    con_upper_orig, con_upper_sidx, con_upper_widx,
    lvar, uvar, lcon, ucon)
  row == nrows || throw(ArgumentError("standard-form row count mismatch"))

  var_start = BoundMap(var_kind, var_idx1, var_idx2, lvar, uvar, Int[])
  con_start = BoundMap(con_kind, con_idx1, con_idx2, lcon, ucon, constraint_rows)
  return StandardFormLayout(nstd, nrows, x0_std, y0_std, rhs, x_offset,
    var_lower, var_upper, var_start, con_start,
    var_upper_eq_rows, con_upper_eq_rows, I, J, V)
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

function _build_standard_linear_data(layout::StandardFormLayout{T}, A_rows, A_cols) where {T}
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
  Astd = sparse(I, J, V, layout.nrows, layout.nstd)
  return LPData(Astd, zeros(T, layout.nstd);
    lcon = layout.rhs, ucon = copy(layout.rhs),
    lvar = zeros(T, layout.nstd), uvar = fill(T(Inf), layout.nstd),
    c0 = zero(T))
end

function _build_standard_quadratic_data(layout::StandardFormLayout{T}, A_rows, A_cols, Q_rows, Q_cols) where {T}
  lp_data = _build_standard_linear_data(layout, A_rows, A_cols)
  vs = layout.var_start
  nnz_q = _count_standard_hessian_entries(Q_rows, Q_cols, vs)
  QI = Vector{Int}(undef, nnz_q); QJ = Vector{Int}(undef, nnz_q); QV = Vector{T}(undef, nnz_q)
  _fill_standard_hessian!(QI, QJ, QV, vs, Q_rows, Q_cols) == nnz_q + 1 || throw(ArgumentError("standard-form Hessian count mismatch"))
  Qstd = sparse(QI, QJ, QV, layout.nstd, layout.nstd)
  return QPData(lp_data.A, zeros(T, layout.nstd), Qstd;
    lcon = lp_data.lcon, ucon = lp_data.ucon,
    lvar = lp_data.lvar, uvar = lp_data.uvar,
    c0 = zero(T), _v = zeros(T, layout.nstd))
end

function _build_scatter_maps(orig, Astd, layout::StandardFormLayout{T}) where {T}
  A_rows, A_cols = _source_structure(orig.data.A)
  vs = layout.var_start

  na = 0
  @inbounds for k in eachindex(A_rows)
    layout.con_start.row[A_rows[k]] == 0 && continue
    na += _standard_var_width(vs.kind[A_cols[k]])
  end
  A_dest = Vector{Int}(undef, na); A_src = Vector{Int}(undef, na); A_scale = Vector{T}(undef, na)
  ka = 1
  @inbounds for k in eachindex(A_rows)
    row = layout.con_start.row[A_rows[k]]
    row == 0 && continue
    _foreach_standard_var(vs, A_cols[k]) do col, scale
      A_dest[ka] = _csc_nz_index(Astd, row, col)
      A_src[ka] = k; A_scale[ka] = scale
      ka += 1
    end
  end
  A_base = copy(SparseArrays.nonzeros(Astd))
  @inbounds for idx in A_dest
    A_base[idx] = zero(T)
  end

  nc = 0
  @inbounds for i in eachindex(vs.kind)
    nc += _standard_var_width(vs.kind[i])
  end
  c_dest = Vector{Int}(undef, nc); c_src = Vector{Int}(undef, nc); c_scale = Vector{T}(undef, nc)
  kc = 1
  @inbounds for i in eachindex(vs.kind)
    _foreach_standard_var(vs, i) do col, scale
      c_dest[kc] = col; c_src[kc] = i; c_scale[kc] = scale
      kc += 1
    end
  end
  c_base = zeros(T, layout.nstd)

  return (A_map = ScatterMap(A_base, A_dest, A_src, A_scale),
          c_map = ScatterMap(c_base, c_dest, c_src, c_scale),
          nnz_src = length(A_rows))
end

_empty_scatter(::Type{T}) where {T} = ScatterMap(T[], Int[], Int[], T[])

function _build_linear_workspace(orig::LinearModel{T}, Astd, layout::StandardFormLayout{T}) where {T}
  maps = _build_scatter_maps(orig, Astd, layout)
  return StandardFormWorkspace(
    orig.data.A, zeros(T, maps.nnz_src), maps.A_map, maps.c_map,
    _structure_signature(orig),
    copy(layout.rhs), layout.x_offset,
    layout.var_start, layout.con_start,
    layout.var_lower, layout.var_upper,
    layout.var_upper_row, layout.con_upper_row,
    zeros(T, NLPModels.get_ncon(orig)), zeros(T, NLPModels.get_ncon(orig)),
    nothing, T[], _empty_scatter(T), T[], T[])
end

function _build_quadratic_workspace(qp::QuadraticModel{T}, std_data::QPData{T}, layout::StandardFormLayout{T}) where {T}
  Astd = std_data.A
  maps = _build_scatter_maps(qp, Astd, layout)
  Q_rows, Q_cols = _source_structure(qp.data.Q)
  vs = layout.var_start

  nq = 0
  @inbounds for k in eachindex(Q_rows)
    nq += _standard_var_width(vs.kind[Q_rows[k]]) * _standard_var_width(vs.kind[Q_cols[k]])
  end
  Q_dest = Vector{Int}(undef, nq); Q_src = Vector{Int}(undef, nq); Q_scale = Vector{T}(undef, nq)
  kq = 1
  @inbounds for k in eachindex(Q_rows)
    _foreach_standard_var(vs, Q_rows[k]) do a, scale_a
      _foreach_standard_var(vs, Q_cols[k]) do b, scale_b
        Q_dest[kq] = _csc_nz_index(std_data.Q, max(a, b), min(a, b))
        Q_src[kq] = k; Q_scale[kq] = scale_a * scale_b
        kq += 1
      end
    end
  end
  Q_base = copy(SparseArrays.nonzeros(std_data.Q))
  @inbounds for idx in Q_dest
    Q_base[idx] = zero(T)
  end

  return StandardFormWorkspace(
    qp.data.A, zeros(T, maps.nnz_src), maps.A_map, maps.c_map,
    _structure_signature(qp),
    copy(layout.rhs), layout.x_offset,
    layout.var_start, layout.con_start,
    layout.var_lower, layout.var_upper,
    layout.var_upper_row, layout.con_upper_row,
    zeros(T, NLPModels.get_ncon(qp)), zeros(T, NLPModels.get_ncon(qp)),
    qp.data.Q, zeros(T, length(Q_rows)),
    ScatterMap(Q_base, Q_dest, Q_src, Q_scale),
    zeros(T, NLPModels.get_nvar(qp)), zeros(T, NLPModels.get_nvar(qp)))
end

# ---------- incremental update ----------

# Per-step CPU kernels; GPU overrides live in the CUDA extension.

function _update_var_start!(xstd::AbstractVector{T}, xsrc::AbstractVector{T}, meta::BoundMap{T}) where {T}
  fill!(xstd, zero(T))
  @inbounds for i in eachindex(meta.kind)
    kind = meta.kind[i]
    if kind == VAR_LB
      xstd[meta.idx1[i]] = xsrc[i] - meta.l[i]
    elseif kind == VAR_LB_UB
      xstd[meta.idx1[i]] = xsrc[i] - meta.l[i]
      xstd[meta.idx2[i]] = meta.u[i] - xsrc[i]
    elseif kind == VAR_UB
      xstd[meta.idx1[i]] = meta.u[i] - xsrc[i]
    elseif kind == VAR_FREE
      xstd[meta.idx1[i]] = max(xsrc[i], zero(T))
      xstd[meta.idx2[i]] = max(-xsrc[i], zero(T))
    end
  end
  return xstd
end

function _update_constraint_start!(xstd::AbstractVector{T}, activity::AbstractVector{T}, meta::BoundMap{T}) where {T}
  @inbounds for i in eachindex(meta.kind)
    kind = meta.kind[i]
    if kind == CON_LB
      xstd[meta.idx1[i]] = activity[i] - meta.l[i]
    elseif kind == CON_RANGE
      xstd[meta.idx1[i]] = activity[i] - meta.l[i]
      xstd[meta.idx2[i]] = meta.u[i] - activity[i]
    elseif kind == CON_UB
      xstd[meta.idx1[i]] = meta.u[i] - activity[i]
    end
  end
  return xstd
end

function _update_x_offset!(x_offset::AbstractVector{T}, meta::BoundMap{T}) where {T}
  @inbounds for i in eachindex(meta.kind)
    kind = meta.kind[i]
    if kind == VAR_LB || kind == VAR_LB_UB
      x_offset[i] = meta.l[i]
    elseif kind == VAR_UB
      x_offset[i] = meta.u[i]
    elseif kind == VAR_FREE
      x_offset[i] = zero(T)
    else
      x_offset[i] = meta.l[i]
    end
  end
  return x_offset
end

function _update_rhs_base!(
  rhs_base::AbstractVector{T},
  var_start::BoundMap{T}, con_start::BoundMap{T},
  var_upper_row::AbstractVector{<:Integer}, con_upper_row::AbstractVector{<:Integer},
) where {T}
  @inbounds for i in eachindex(con_start.row)
    row = con_start.row[i]
    row == 0 && continue
    rhs_base[row] = isfinite(con_start.l[i]) ? con_start.l[i] : con_start.u[i]
  end
  @inbounds for i in eachindex(var_upper_row)
    row = var_upper_row[i]
    row > 0 && (rhs_base[row] = var_start.u[i] - var_start.l[i])
  end
  @inbounds for i in eachindex(con_upper_row)
    row = con_upper_row[i]
    row > 0 && (rhs_base[row] = con_start.u[i] - con_start.l[i])
  end
  return rhs_base
end

function _update_dual_start!(ystd::AbstractVector{T}, ysrc::AbstractVector{T}, rows::AbstractVector{<:Integer}) where {T}
  fill!(ystd, zero(T))
  @inbounds for i in eachindex(rows)
    row = rows[i]
    row > 0 && (ystd[row] = ysrc[i])
  end
  return ystd
end

function _apply_rhs_shift!(rhs::AbstractVector{T}, rhs_base::AbstractVector{T}, shift::AbstractVector{T}, rows::AbstractVector{<:Integer}) where {T}
  copyto!(rhs, rhs_base)
  @inbounds for i in eachindex(rows)
    row = rows[i]
    row > 0 && (rhs[row] -= shift[i])
  end
  return rhs
end

_mul_sparse_symmetric!(dest, Q::SparseMatrixCSC, x) = mul!(dest, Symmetric(Q, :L), x)
_mul_sparse_symmetric!(dest, Q, x) = mul!(dest, Q, x)

# Dirty flags: which user-facing fields changed since the last refresh.
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

function _apply!(std, ws::StandardFormWorkspace{T}, src, d::_Dirty) where {T}
  # Derived flags: each names the standard-form quantity it gates.
  rhs_base_dirty  = d.var_bounds || d.con_bounds
  shift_dirty     = d.A          || d.var_bounds
  rhs_dirty       = rhs_base_dirty || shift_dirty
  x0_var_dirty    = d.var_bounds || d.x0
  activity_dirty  = d.A          || d.x0
  x0_slack_dirty  = activity_dirty || d.con_bounds
  qx_dirty        = d.Q          || d.var_bounds

  # 1. Sync bound metadata from src and derived x_offset.
  if d.var_bounds
    copyto!(ws.var_start.l, src.meta.lvar)
    copyto!(ws.var_start.u, src.meta.uvar)
    _update_x_offset!(ws.x_offset, ws.var_start)
  end
  if d.con_bounds
    copyto!(ws.con_start.l, src.meta.lcon)
    copyto!(ws.con_start.u, src.meta.ucon)
  end

  # 2. Scatter original A into standard-form A.
  if d.A
    _copy_sparse_values!(ws.A_ref, ws.A_src)
    _apply_scatter_map!(_sparse_nzvals(std.data.A), ws.A_map, ws.A_src)
  end

  # 3. Standard-form RHS: std.lcon = std.ucon = rhs_base - A * x_offset.
  if rhs_base_dirty
    _update_rhs_base!(ws.rhs_base, ws.var_start, ws.con_start, ws.var_upper_row, ws.con_upper_row)
  end
  if shift_dirty
    mul!(ws.shift, ws.A_ref, ws.x_offset)
  end
  if rhs_dirty
    _apply_rhs_shift!(std.data.lcon, ws.rhs_base, ws.shift, ws.con_start.row)
    copyto!(std.data.ucon, std.data.lcon)
  end

  # 4. Initial iterate (x0, y0) in standard space.
  if x0_var_dirty
    _update_var_start!(std.meta.x0, src.meta.x0, ws.var_start)
  end
  if activity_dirty
    mul!(ws.activity, ws.A_ref, src.meta.x0)
  end
  if x0_slack_dirty
    _update_constraint_start!(std.meta.x0, ws.activity, ws.con_start)
  end
  if d.y0 || d.con_bounds
    _update_dual_start!(std.meta.y0, src.meta.y0, ws.con_start.row)
  end

  # 5. Objective in standard space (linear path + quadratic correction).
  if ws.Q_ref === nothing
    _apply_lp_objective!(std, ws, src, d)
  else
    _apply_qp_objective!(std, ws, src, d, qx_dirty)
  end
  return std
end

function _apply_lp_objective!(std, ws, src, d::_Dirty)
  if d.c
    _apply_scatter_map!(std.data.c, ws.c_map, src.data.c)
  end
  if d.c0 || d.c || d.var_bounds
    std.data.c0[] = src.data.c0[] + dot(src.data.c, ws.x_offset)
  end
  return
end

function _apply_qp_objective!(std, ws, src, d::_Dirty, qx_dirty::Bool)
  if d.Q
    _copy_sparse_values!(ws.Q_ref, ws.Q_src)
    _apply_scatter_map!(_sparse_nzvals(std.data.Q), ws.Q_map, ws.Q_src)
  end
  if qx_dirty
    _mul_sparse_symmetric!(ws.qx, ws.Q_ref, ws.x_offset)
  end
  # Std linear cost = original c shifted by Q * x_offset, then scattered.
  if d.c || qx_dirty
    copyto!(ws.ctmp, src.data.c)
    ws.ctmp .+= ws.qx
    _apply_scatter_map!(std.data.c, ws.c_map, ws.ctmp)
  end
  if d.c0 || d.c || qx_dirty
    std.data.c0[] = src.data.c0[] + dot(src.data.c, ws.x_offset) + dot(ws.qx, ws.x_offset) / 2
  end
  return
end

# ---------- public build / update API ----------

"""
    std, ws = standard_form(orig::LinearModel | ::QuadraticModel)

Reformulate `orig` into standard form. Returns the standard-form model `std`
and a [`StandardFormWorkspace`](@ref) that caches the mapping so that value
updates on `orig` can be pushed through `update_standard_form!` without
rebuilding.
"""
standard_form(lp::LinearModel) = _build_standard_linear(lp)
standard_form(qp::QuadraticModel) = _build_standard_quadratic(qp)

function _build_standard_linear(lp::LinearModel)
  layout = _build_standard_layout(lp)
  A_rows, A_cols = _source_structure(lp.data.A)
  data = _build_standard_linear_data(layout, A_rows, A_cols)
  isempty(data.c) && throw(ArgumentError("Standard-form reformulation eliminated all decision variables; trivial all-fixed models are not supported."))
  std = LinearModel(data; x0 = layout.x0, y0 = layout.y0, minimize = lp.meta.minimize, name = lp.meta.name)
  ws = _build_linear_workspace(lp, std.data.A, layout)
  update_standard_form!(lp, std, ws)
  return std, ws
end

function _build_standard_quadratic(qp::QuadraticModel)
  layout = _build_standard_layout(qp)
  A_rows, A_cols = _source_structure(qp.data.A)
  Q_rows, Q_cols = _source_structure(qp.data.Q)
  data = _build_standard_quadratic_data(layout, A_rows, A_cols, Q_rows, Q_cols)
  isempty(data.c) && throw(ArgumentError("Standard-form reformulation eliminated all decision variables; trivial all-fixed models are not supported."))
  std = QuadraticModel(data; x0 = layout.x0, y0 = layout.y0, minimize = qp.meta.minimize, name = qp.meta.name)
  ws = _build_quadratic_workspace(qp, std.data, layout)
  update_standard_form!(qp, std, ws)
  return std, ws
end

"""
    update_standard_form!(orig, std, ws; c, c0, A, Q, lvar, uvar, lcon, ucon, x0, y0)

Mutate the original model `orig` with any provided fields (all in the
original variable/constraint space), then propagate the minimal set of
updates to the standard-form model `std`. When called with no keyword
arguments, performs a full refresh from the current state of `orig` — useful
after mutating `orig` directly.

Sparsity patterns and bound kinds (finite ↔ infinite, `l == u`) must be
unchanged; for structural changes, rebuild via [`standard_form`](@ref).
Sparse matrices passed via `A` or `Q` are copied by nonzero values.

Returns `std`.
"""
function update_standard_form!(orig, std, ws::StandardFormWorkspace;
                               c = nothing, c0 = nothing, A = nothing, Q = nothing,
                               lvar = nothing, uvar = nothing, lcon = nothing, ucon = nothing,
                               x0 = nothing, y0 = nothing)
  c      === nothing || copyto!(orig.data.c, c)
  c0     === nothing || (orig.data.c0[] = c0)
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
  # Any change that could alter a bound kind (finite ↔ infinite, l == u) or the
  # sparsity pattern invalidates the standard-form structure; force a rebuild.
  if (var_bounds_changed || con_bounds_changed) && _structure_signature(orig) != ws.signature
    throw(ArgumentError(
      "update_standard_form! cannot absorb changes that alter bound kinds (finite ↔ infinite, l == u) or the sparsity pattern; rebuild with standard_form(orig)",
    ))
  end
  # No kwargs => full refresh (caller mutated orig externally).
  d = any_given ? _Dirty(
    c          = c    !== nothing,
    c0         = c0   !== nothing,
    A          = A    !== nothing,
    Q          = Q    !== nothing,
    var_bounds = var_bounds_changed,
    con_bounds = con_bounds_changed,
    x0         = x0   !== nothing,
    y0         = y0   !== nothing,
  ) : _ALL_DIRTY
  _apply!(std, ws, orig, d)
  return std
end
