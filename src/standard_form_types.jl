"""
    ScatterMap(base, dest, src, scale)

Linear scatter: `dest .= base; dest[map.dest[k]] += scale[k] * src[map.src[k]]`
for each `k`. `base` carries destination entries that don't depend on `src`.
"""
struct ScatterMap{T, VT <: AbstractVector{T}, VI <: AbstractVector{Int}}
  base::VT
  dest::VI
  src::VI
  scale::VT
end

"""
    BoundMap{T,LU,VI,VU}

Per-row/column bound metadata. `kind[i]` is the bound kind; `idx1`/`idx2` are
the std-form slot(s) that orig entry `i` maps to; `l`/`u` hold the current
bounds. `row` is populated for constraint maps, empty for variable maps. `LU`
is `Vector` for the scalar path, `Matrix` for batch.
"""
struct BoundMap{T, LU <: Union{AbstractVector{T}, AbstractMatrix{T}},
                VI <: AbstractVector{Int}, VU <: AbstractVector{BoundKind}}
  kind::VU
  idx1::VI
  idx2::VI
  l::LU
  u::LU
  row::VI
end

"""
    StandardFormWorkspace

Caches the mapping from a [`LinearModel`](@ref)/[`QuadraticModel`](@ref) or
[`BatchQuadraticModel`](@ref) to its standard-form image.

The `IT <: AbstractVecOrMat{T}` parameter selects the iterate kind:
`Vector{T}` for scalar models, `Matrix{T}` for batch models. `Q_ref === nothing`
flags an LP and disables the Hessian branch of the update; `c0_batch`/`c0_tmp`
have length 0 for scalar workspaces (the c0 destination is `std.data.c0[]`)
and length `nbatch` for batch workspaces.
"""
struct StandardFormWorkspace{T,
                             IT <: AbstractVecOrMat{T}, VT <: AbstractVector{T},
                             VI <: AbstractVector{Int}, VU <: AbstractVector{BoundKind},
                             ARef, QRef}
  A_ref::ARef
  Q_ref::QRef
  A_map::ScatterMap{T, VT, VI}
  Q_map::ScatterMap{T, VT, VI}
  c_map::ScatterMap{T, VT, VI}
  var_start::BoundMap{T, IT, VI, VU}
  con_start::BoundMap{T, IT, VI, VU}
  var_lower::VI
  var_upper::VI
  var_upper_row::VI
  con_upper_row::VI
  rhs_base::IT
  x_offset::IT
  shift::IT
  activity::IT
  qx::IT
  ctmp::IT
  c0_batch::VT
  c0_tmp::VT
end

Adapt.@adapt_structure ScatterMap
Adapt.@adapt_structure BoundMap
Adapt.@adapt_structure StandardFormWorkspace
