# GPU-resident `standard_form` build for scalar models. Avoids host pulls of
# device-resident `_sparse_structure` outputs by emitting the std-form
# Jacobian/Hessian and scatter maps entirely on device.

function _device_total(prefix::CuVector{<:Integer})
  isempty(prefix) && return 0
  CUDA.@allowscalar return Int(prefix[end])
end

function _scan_counts(counts::CuVector{Int})
  prefix = similar(counts)
  isempty(counts) || CUDA.scan!(+, prefix, counts; dims = 1)
  return prefix
end

# Exclusive prefix scan with terminating total: `out[i] = sum(counts[1:i-1])`,
# `out[end] = sum(counts)`. Length is `length(counts) + 1`.
function _exclusive_prefix(counts::CuVector{Int})
  n = length(counts)
  out = CUDA.zeros(Int, n + 1)
  n > 0 && CUDA.scan!(+, view(out, 2:n+1), counts; dims = 1)
  return out
end

# Pull the totals (final element of each prefix) in a single bulk transfer.
function _read_totals(prefixes::Vararg{CuVector{Int}, N}) where {N}
  totals = CuVector{Int}(undef, N)
  @inbounds for k in 1:N
    p = prefixes[k]
    copyto!(view(totals, k:k), view(p, length(p):length(p)))
  end
  return Vector(totals)
end

@kernel function _jacobian_counts_kernel!(counts, rows, cols, constraint_rows, x_kind)
  k = @index(Global, Linear)
  @inbounds if k <= length(rows)
    row = constraint_rows[rows[k]]
    counts[k] = row == 0 ? 0 : _standard_var_width(x_kind[cols[k]])
  end
end

@inline function _var_expand(kind::BoundKind, idx1::Int, idx2::Int, ::Type{T}) where {T}
  if kind == VAR_LB || kind == VAR_LB_UB
    return (idx1, 0), (one(T), zero(T)), 1
  elseif kind == VAR_UB
    return (idx1, 0), (-one(T), zero(T)), 1
  elseif kind == VAR_FREE
    return (idx1, idx2), (one(T), -one(T)), 2
  else
    return (0, 0), (zero(T), zero(T)), 0
  end
end

@kernel function _jacobian_fill_kernel!(I, J, V, dest, src, scale, counts, prefix,
                                         rows, cols, constraint_rows, x_kind, x_idx1, x_idx2,
                                         matrix_offset)
  k = @index(Global, Linear)
  T = eltype(V)
  @inbounds if k <= length(rows) && counts[k] != 0
    count = counts[k]
    row = constraint_rows[rows[k]]; col = cols[k]
    first = prefix[k] - count + 1
    idxs, scales, cnt = _var_expand(x_kind[col], x_idx1[col], x_idx2[col], T)
    @inbounds for a in 1:cnt
      pos = matrix_offset + first + (a - 1)
      I[pos] = row; J[pos] = idxs[a]; V[pos] = scales[a]
      dest[first + (a - 1)]  = pos
      src[first + (a - 1)]   = k
      scale[first + (a - 1)] = scales[a]
    end
  end
end

@kernel function _hessian_counts_kernel!(counts, rows, cols, x_kind)
  k = @index(Global, Linear)
  @inbounds if k <= length(rows)
    counts[k] = _standard_var_width(x_kind[rows[k]]) * _standard_var_width(x_kind[cols[k]])
  end
end

@kernel function _hessian_fill_kernel!(I, J, V, dest, src, scale, counts, prefix,
                                        rows, cols, x_kind, x_idx1, x_idx2)
  k = @index(Global, Linear)
  T = eltype(V)
  @inbounds if k <= length(rows) && counts[k] != 0
    count = counts[k]
    i = rows[k]; j = cols[k]
    start = prefix[k] - count + 1
    idxs_i, scales_i, ci = _var_expand(x_kind[i], x_idx1[i], x_idx2[i], T)
    idxs_j, scales_j, cj = _var_expand(x_kind[j], x_idx1[j], x_idx2[j], T)
    idx = start
    @inbounds for a in 1:ci, b in 1:cj
      ai, aj = idxs_i[a], idxs_j[b]
      sab = scales_i[a] * scales_j[b]
      I[idx] = max(ai, aj); J[idx] = min(ai, aj); V[idx] = sab
      dest[idx] = idx; src[idx] = k; scale[idx] = sab
      idx += 1
    end
  end
end

# Device-resident c-map (mirrors CPU `_build_c_map`); avoids CPU iteration over
# the now-device-resident `var_start.kind`.
function _build_c_map_device(layout)
  T = eltype(layout.x_offset)
  backend = CUDABackend()
  vs = layout.var_start
  n = length(vs.kind)
  count = CuVector{Int}(undef, n)
  n > 0 && _cmap_count_kernel!(backend)(count, vs.kind; ndrange = n)
  prefix = _exclusive_prefix(count)
  total = Vector(view(prefix, n+1:n+1))[1]
  dest = CuVector{Int}(undef, total); src = CuVector{Int}(undef, total); scale = CuVector{T}(undef, total)
  total > 0 && _cmap_fill_kernel!(backend)(dest, src, scale, vs.kind, vs.idx1, vs.idx2, prefix; ndrange = n)
  return ScatterMap(CUDA.zeros(T, layout.nstd), dest, src, scale)
end

# Device-resident `_build_standard_layout`: classification → prefix scans →
# fill kernels. All metadata stays on device; the only host transfer is one
# bulk read of five small totals to size the output arrays.
function _build_standard_layout(model::Union{
    LinearModel{T, <:CuVector{T}, <:AbstractSparseOperator{T}},
    QuadraticModel{T, <:CuVector{T}, <:CuVector{T}, <:AbstractSparseOperator{T}, <:AbstractSparseOperator{T}},
}) where {T}
  n = NLPModels.get_nvar(model); m = NLPModels.get_ncon(model)
  backend = CUDABackend()
  lvar = model.meta.lvar; uvar = model.meta.uvar
  lcon = model.meta.lcon; ucon = model.meta.ucon
  x0_src = model.meta.x0;  y0_src = model.meta.y0

  # Classification.
  var_kind  = CuVector{BoundKind}(undef, n)
  var_width = CuVector{Int}(undef, n)
  var_has_upper = CuVector{Int}(undef, n)
  n > 0 && _classify_var_kernel!(backend)(var_kind, var_width, var_has_upper, lvar, uvar; ndrange = n)

  con_kind = CuVector{BoundKind}(undef, m)
  con_kept = CuVector{Int}(undef, m)
  con_slack_width = CuVector{Int}(undef, m)
  con_has_range = CuVector{Int}(undef, m)
  m > 0 && _classify_con_kernel!(backend)(con_kind, con_kept, con_slack_width, con_has_range, lcon, ucon; ndrange = m)

  # Exclusive prefix scans (length n+1 / m+1; last entry holds total).
  var_prefix    = _exclusive_prefix(var_width)
  kept_prefix   = _exclusive_prefix(con_kept)
  slack_prefix  = _exclusive_prefix(con_slack_width)
  var_eq_prefix = _exclusive_prefix(var_has_upper)
  con_eq_prefix = _exclusive_prefix(con_has_range)

  # Single GPU→CPU bulk read of all five totals.
  totals = _read_totals(var_prefix, slack_prefix, kept_prefix, var_eq_prefix, con_eq_prefix)
  nstd_v, nstd_s, nrows_p, n_var_eq, n_con_eq = totals
  nstd  = nstd_v + nstd_s
  nrows = nrows_p + n_var_eq + n_con_eq
  n_extra = nstd_s + 2 * (n_var_eq + n_con_eq)

  # Output allocations.
  var_idx1 = CUDA.zeros(Int, n);  var_idx2 = CUDA.zeros(Int, n)
  var_lower = CUDA.zeros(Int, n); var_upper = CUDA.zeros(Int, n)
  x_offset = CuVector{T}(undef, n)
  x0_std = CUDA.zeros(T, nstd)
  con_idx1 = CUDA.zeros(Int, m); con_idx2 = CUDA.zeros(Int, m)
  constraint_rows = CUDA.zeros(Int, m)
  rhs = CUDA.zeros(T, nrows); y0_std = CUDA.zeros(T, nrows)
  extra_I = CuVector{Int}(undef, n_extra); extra_J = CuVector{Int}(undef, n_extra)
  extra_V = CuVector{T}(undef, n_extra)
  var_upper_row = CUDA.zeros(Int, n); con_upper_row = CUDA.zeros(Int, m)

  # Fill.
  n > 0 && _layout_var_kernel!(backend)(var_idx1, var_idx2, var_lower, var_upper, x_offset, x0_std,
    var_kind, var_prefix, lvar, uvar, x0_src; ndrange = n)
  m > 0 && _layout_con_kernel!(backend)(con_idx1, con_idx2, constraint_rows,
    rhs, y0_std, extra_I, extra_J, extra_V,
    con_kind, kept_prefix, slack_prefix, lcon, ucon, y0_src, nstd_v; ndrange = m)
  n_var_eq > 0 && _layout_eq_kernel!(backend)(var_upper_row, extra_I, extra_J, extra_V, rhs,
    var_has_upper, var_eq_prefix, var_idx1, var_idx2, lvar, uvar,
    nstd_s, nrows_p; ndrange = n)
  n_con_eq > 0 && _layout_eq_kernel!(backend)(con_upper_row, extra_I, extra_J, extra_V, rhs,
    con_has_range, con_eq_prefix, con_idx1, con_idx2, lcon, ucon,
    nstd_s + 2 * n_var_eq, nrows_p + n_var_eq; ndrange = m)

  empty_int = similar(var_kind, Int, 0)
  return (
    nstd = nstd, nrows = nrows,
    x0 = x0_std, y0 = y0_std, rhs = rhs, x_offset = x_offset,
    var_lower = var_lower, var_upper = var_upper,
    var_start = BoundMap(var_kind, var_idx1, var_idx2, lvar, uvar, empty_int),
    con_start = BoundMap(con_kind, con_idx1, con_idx2, lcon, ucon, constraint_rows),
    var_upper_row = var_upper_row,
    con_upper_row = con_upper_row,
    extra_I = extra_I, extra_J = extra_J, extra_V = extra_V,
  )
end

# CUSPARSE SpMV on an unsorted COO silently produces wrong results. The fill
# kernels emit in orig-k order, so convert to CSR and carry the COO→CSR perm
# through the scatter map.
@kernel function _inverse_perm_kernel!(perm, marker)
  i = @index(Global, Linear)
  @inbounds i <= length(marker) && (perm[Int(marker[i])] = i)
end

@kernel function _remap_dest_kernel!(dest, perm)
  i = @index(Global, Linear)
  @inbounds i <= length(dest) && (dest[i] = perm[dest[i]])
end

@kernel function _permute_base_kernel!(base_new, base_old, perm)
  i = @index(Global, Linear)
  @inbounds i <= length(base_old) && (base_new[perm[i]] = base_old[i])
end

function _coo_to_csr_remap(coo::CUSPARSE.CuSparseMatrixCOO{T},
                            base::CuVector{T}, dest::CuVector{Int}) where {T}
  nnz_tot = length(base)
  val_csr = CUSPARSE.CuSparseMatrixCSR(coo)
  nnz_tot == 0 && return val_csr, base
  backend = CUDABackend()
  marker_coo = CUSPARSE.CuSparseMatrixCOO(
    copy(coo.rowInd), copy(coo.colInd), CuArray(T.(1:nnz_tot)), size(coo))
  marker_csr = CUSPARSE.CuSparseMatrixCSR(marker_coo)
  perm = CuVector{Int}(undef, nnz_tot)
  _inverse_perm_kernel!(backend)(perm, marker_csr.nzVal; ndrange = nnz_tot)
  length(dest) > 0 && _remap_dest_kernel!(backend)(dest, perm; ndrange = length(dest))
  base_csr = CUDA.zeros(T, nnz_tot)
  _permute_base_kernel!(backend)(base_csr, base, perm; ndrange = nnz_tot)
  return val_csr, base_csr
end

function _build_device_jacobian(layout, rows::CuVector{Int}, cols::CuVector{Int})
  T = eltype(layout.x_offset)
  backend = CUDABackend()
  constraint_rows = layout.con_start.row
  x_kind = layout.var_start.kind

  counts = CuVector{Int}(undef, length(rows))
  isempty(rows) || _jacobian_counts_kernel!(backend)(counts, rows, cols, constraint_rows, x_kind; ndrange = length(rows))
  prefix = _scan_counts(counts)
  source_nnz = _device_total(prefix)
  extra_nnz = length(layout.extra_I)
  total_nnz = extra_nnz + source_nnz

  I = CuVector{Int}(undef, total_nnz); J = similar(I); V = CuVector{T}(undef, total_nnz)
  base = CUDA.zeros(T, total_nnz)
  if extra_nnz > 0
    copyto!(view(I, 1:extra_nnz), layout.extra_I)
    copyto!(view(J, 1:extra_nnz), layout.extra_J)
    copyto!(view(V, 1:extra_nnz), layout.extra_V)
    copyto!(view(base, 1:extra_nnz), layout.extra_V)
  end

  dest = CuVector{Int}(undef, source_nnz); src = similar(dest); scale = CuVector{T}(undef, source_nnz)
  if source_nnz > 0
    _jacobian_fill_kernel!(backend)(
      I, J, V, dest, src, scale, counts, prefix, rows, cols,
      constraint_rows, x_kind, layout.var_start.idx1, layout.var_start.idx2, extra_nnz; ndrange = length(rows))
  end
  coo = CUSPARSE.CuSparseMatrixCOO(I, J, V, (layout.nrows, layout.nstd))
  csr, base_csr = _coo_to_csr_remap(coo, base, dest)
  return csr, ScatterMap(base_csr, dest, src, scale)
end

function _build_device_hessian(layout, rows::CuVector{Int}, cols::CuVector{Int})
  T = eltype(layout.x_offset)
  backend = CUDABackend()
  x_kind = layout.var_start.kind

  counts = CuVector{Int}(undef, length(rows))
  isempty(rows) || _hessian_counts_kernel!(backend)(counts, rows, cols, x_kind; ndrange = length(rows))
  prefix = _scan_counts(counts)
  total_nnz = _device_total(prefix)
  I = CuVector{Int}(undef, total_nnz); J = similar(I); V = CuVector{T}(undef, total_nnz)
  dest = CuVector{Int}(undef, total_nnz); src = similar(dest); scale = CuVector{T}(undef, total_nnz)
  if total_nnz > 0
    _hessian_fill_kernel!(backend)(
      I, J, V, dest, src, scale, counts, prefix, rows, cols,
      x_kind, layout.var_start.idx1, layout.var_start.idx2; ndrange = length(rows))
  end
  coo = CUSPARSE.CuSparseMatrixCOO(I, J, V, (layout.nstd, layout.nstd))
  base = CUDA.zeros(T, total_nnz)
  csr, base_csr = _coo_to_csr_remap(coo, base, dest)
  return csr, ScatterMap(base_csr, dest, src, scale)
end

function _device_workspace(orig, layout, A_map, Q_ref, Q_map)
  T = eltype(layout.x_offset)
  y_template = NLPModels.get_y0(orig)
  if Q_ref !== nothing
    x_template = NLPModels.get_x0(orig)
    qx = similar(x_template); ctmp = similar(x_template)
  else
    Q_map = ScatterMap(CUDA.zeros(T, 0), CUDA.zeros(Int, 0), CUDA.zeros(Int, 0), CUDA.zeros(T, 0))
    qx = CUDA.zeros(T, 0); ctmp = CUDA.zeros(T, 0)
  end
  return StandardFormWorkspace(
    orig.data.A, Q_ref,
    A_map, Q_map, _build_c_map_device(layout),
    layout.var_start, layout.con_start,
    layout.var_lower, layout.var_upper,
    layout.var_upper_row, layout.con_upper_row,
    layout.rhs, layout.x_offset,
    similar(y_template), similar(y_template),
    qx, ctmp,
    CUDA.zeros(T, 0), CUDA.zeros(T, 0),
  )
end

# Direct GPU build entry point (LP and QP).
function standard_form(orig::Union{
    LinearModel{T, <:CuVector{T}, <:AbstractSparseOperator{T}},
    QuadraticModel{T, <:CuVector{T}, <:CuVector{T}, <:AbstractSparseOperator{T}, <:AbstractSparseOperator{T}},
}) where {T}
  layout = _build_standard_layout(orig)
  A_rows, A_cols = _sparse_structure(orig.data.A)
  Astd, A_map = _build_device_jacobian(layout, A_rows, A_cols)
  Astd_op = sparse_operator(Astd)
  nstd = layout.nstd
  c = CUDA.zeros(T, nstd)
  bounds = (lcon = layout.rhs, ucon = layout.rhs,
            lvar = CUDA.zeros(T, nstd), uvar = CUDA.fill(T(Inf), nstd))
  meta = (x0 = layout.x0, y0 = layout.y0,
          minimize = orig.meta.minimize, name = orig.meta.name)

  if orig isa QuadraticModel
    Q_rows, Q_cols = _sparse_structure(orig.data.Q)
    Qstd, Q_map = _build_device_hessian(layout, Q_rows, Q_cols)
    data = QPData(Astd_op, c, sparse_operator(Qstd; symmetric = true);
                  bounds..., c0 = zero(T), _v = CUDA.zeros(T, nstd))
    std = QuadraticModel(data; meta...)
    ws = _device_workspace(orig, layout, A_map, orig.data.Q, Q_map)
  else
    data = LPData(Astd_op, c; bounds..., c0 = zero(T))
    std = LinearModel(data; meta...)
    ws = _device_workspace(orig, layout, A_map, nothing, nothing)
  end
  update_standard_form!(orig, std, ws)
  return std, ws
end

# Batch path: rebuild GPU sparse ops with `spmm_ncols = nbatch` so the batched
# SpMM buffer is premade for the batched standard-form path.
@inline _spmm_op(op, nbatch; symmetric::Bool) =
  sparse_operator(operator_sparse_matrix(op); symmetric, spmm_ncols = nbatch)

function _adapt_to_batch_backend(qp::QuadraticModel, ::Type{<:CuMatrix}, nbatch::Int)
  gpu = Adapt.adapt(CuArray, qp); data = gpu.data
  rebuilt = QPData(_spmm_op(data.A, nbatch; symmetric = false), data.c,
                   _spmm_op(data.Q, nbatch; symmetric = true);
    lcon = data.lcon, ucon = data.ucon, lvar = data.lvar, uvar = data.uvar,
    c0 = data.c0, _v = data._v)
  return QuadraticModel(rebuilt;
    x0 = gpu.meta.x0, y0 = gpu.meta.y0, minimize = gpu.meta.minimize, name = gpu.meta.name)
end
