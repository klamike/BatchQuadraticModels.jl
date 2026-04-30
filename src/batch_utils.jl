# Helpers shared by the batch constructors and the standard-form presolve.

# ---------- column stacking / replication ----------

# Stack per-instance vectors side-by-side into a `(dim, nbatch)` matrix on
# backend `MT`. `getter` projects each element of `src` to a vector (default
# `identity`, e.g. pass `qp -> qp.meta.lvar` to stack a model field).
function _stack_columns(MT, src, getter = identity)
  @assert !isempty(src) "Need at least one column"
  out = MT(undef, length(getter(first(src))), length(src))
  for (j, item) in enumerate(src)
    copyto!(view(out, :, j), getter(item))
  end
  return out
end

# Broadcast a single vector across all `nbatch` columns of a fresh `MT` matrix.
function _repeat_column(MT, col, nbatch)
  out = MT(undef, length(col), nbatch)
  out .= col
  return out
end

# Stack the six `NLPModelMeta` bound/iterate fields into batch matrices.
function _stack_batch_bounds(MT, qps)
  return (
    _stack_columns(MT, qps, qp -> qp.meta.x0),
    _stack_columns(MT, qps, qp -> qp.meta.y0),
    _stack_columns(MT, qps, qp -> qp.meta.lvar),
    _stack_columns(MT, qps, qp -> qp.meta.uvar),
    _stack_columns(MT, qps, qp -> qp.meta.lcon),
    _stack_columns(MT, qps, qp -> qp.meta.ucon),
  )
end

# Stack per-instance scalar objective constants into a batch vector.
function _stack_c0(qps::AbstractVector, ::Type{T}) where {T}
  c0 = similar(qps[1].data.c, T, length(qps))
  @inbounds for (j, qp) in enumerate(qps)
    c0[j] = qp.data.c0[]
  end
  return c0
end

# Keeps index arrays on the same backend as `template` (CPU → Vector, GPU →
# CuVector) so downstream dispatch stays device-resident.
@inline _indices_like(template::AbstractVector, n::Int) =
    (out = similar(template, Int, n); out .= 1:n; out)

# ---------- BatchNLPModelMeta construction / adapt ----------

# Build a `BatchNLPModelMeta` carrying `nbatch` copies of each bound/iterate
# vector; `ncon`/`nnzj`/`minimize` inherit from the source scalar `meta`.
function _batch_meta(::Type{T}, ::Type{MT}, meta, nbatch; x0, y0 = fill!(MT(undef, meta.ncon, nbatch), zero(T)), lvar, uvar, lcon, ucon, nnzh = meta.nnzh, islp = meta.islp, name = meta.name) where {T, MT}
  return NLPModels.BatchNLPModelMeta{T, MT}(nbatch, meta.nvar;
    x0, lvar, uvar,
    ncon = meta.ncon, y0, lcon, ucon,
    nnzj = meta.nnzj, nnzh,
    minimize = meta.minimize, islp, name,
  )
end

# Drive `Adapt.adapt_structure(BatchQuadraticModel)` — the batch meta needs a
# manual rebuild because `NLPModels.BatchNLPModelMeta` has no Adapt support.
function _adapt_batch_meta(to, meta::NLPModels.BatchNLPModelMeta{T}) where {T}
  x0 = Adapt.adapt(to, meta.x0)
  return _batch_meta(T, typeof(x0), meta, meta.nbatch;
    x0,
    y0 = Adapt.adapt(to, meta.y0),
    lvar = Adapt.adapt(to, meta.lvar),
    uvar = Adapt.adapt(to, meta.uvar),
    lcon = Adapt.adapt(to, meta.lcon),
    ucon = Adapt.adapt(to, meta.ucon),
  )
end
