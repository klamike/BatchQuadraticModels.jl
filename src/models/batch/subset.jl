# Subset evaluation: each `*_subset!` takes a `roots` vector of the active
# batch columns (length `na`) and evaluates the operation over those columns
# only. Skipping inactive instances lets the batch IPM avoid wasted work as
# instances converge and get masked out.
#
# Dispatch pattern per function: shared-matrix variant (scalar `SparseOperator`)
# forwards to the full-batch `NLPModels` method since the per-column work is
# already packed; per-instance variant (`BatchSparseOperator`) takes the subset
# path through `batch_spmv_subset!` to skip masked-out columns in the SpMV.
#
# `bqp._HX`/`bqp._CX` are scratch matrices sized `(nvar, nbatch)`; we view the
# first `length(roots)` columns of each for the subset-size workspace.

# ---------- per-kind gather helpers ----------

@inline _gather_c!(dest::AbstractMatrix, c::AbstractVector, roots) = (dest .= c; dest)
@inline _gather_c!(dest::AbstractMatrix, c::AbstractMatrix, roots) = gather_columns!(dest, c, roots)

@inline _scatter_c0_subset!(bf, c0::Number,         roots) = (bf .+= c0)
@inline _scatter_c0_subset!(bf, c0::AbstractVector, roots) = (bf .+= view(c0, roots))

# ---------- obj_subset! ----------

"""
    obj_subset!(bqp, bx, bf, roots)

Batched objective evaluation restricted to active instances. Computes
`bf[j] = (1/2) bx[:,j]'Q bx[:,j] + c'bx[:,j] + c0` for each `j in 1:length(roots)`.
"""
function obj_subset!(
  bqp::BatchQuadraticModel{T},
  bx::AbstractMatrix, bf::AbstractVector, roots::AbstractVector{<:Integer},
) where {T}
  na = length(roots)
  HX = view(bqp._HX, :, 1:na)
  _obj_subset_accumulate_HX!(HX, bqp, bx, roots)
  batch_mapreduce!(*, +, zero(T), reshape(bf, 1, na), HX, bx)
  _scatter_c0_subset!(bf, bqp.c0_batch, roots)
  return bf
end

# Shared Q: scalar SpMV on all columns, then gather c at `roots`.
function _obj_subset_accumulate_HX!(HX, bqp::BatchQuadraticModel{T, MT, VT, AOp, <:AbstractSparseOperator}, bx, roots) where {T, MT, VT, AOp}
  mul!(HX, bqp.Q, bx, T(0.5), zero(T))
  ctmp = view(bqp._CX, :, 1:length(roots))
  _gather_c!(ctmp, bqp.c_batch, roots)
  HX .+= ctmp
  return HX
end

# Varying Q: subset SpMV skips inactive instances entirely.
function _obj_subset_accumulate_HX!(HX, bqp::BatchQuadraticModel{T, MT, VT, AOp, <:BatchSparseOperator}, bx, roots) where {T, MT, VT, AOp}
  _gather_c!(HX, bqp.c_batch, roots)
  batch_spmv_subset!(HX, bqp.Q, bx, roots, T(0.5), one(T))
  return HX
end

# ---------- grad_subset! ----------

"""
    grad_subset!(bqp, bx, bg, roots)

Batched gradient `bg[:, j] = Q bx[:, j] + c` for each active instance `j`.
"""
function grad_subset!(bqp::BatchQuadraticModel{T, MT, VT, AOp, <:AbstractSparseOperator}, bx::AbstractMatrix, bg::AbstractMatrix, roots::AbstractVector{<:Integer}) where {T, MT, VT, AOp}
  HX = view(bqp._HX, :, 1:length(roots))
  mul!(HX, bqp.Q, bx)
  _gather_c!(bg, bqp.c_batch, roots)
  bg .+= HX
  return bg
end

function grad_subset!(bqp::BatchQuadraticModel{T, MT, VT, AOp, <:BatchSparseOperator}, bx::AbstractMatrix, bg::AbstractMatrix, roots::AbstractVector{<:Integer}) where {T, MT, VT, AOp}
  _gather_c!(bg, bqp.c_batch, roots)
  batch_spmv_subset!(bg, bqp.Q, bx, roots, one(T), one(T))
  return bg
end

# ---------- cons_subset! ----------

"""
    cons_subset!(bqp, bx, bc, roots)

Batched constraint residual `bc[:, j] = A bx[:, j]` for each active instance `j`.
"""
cons_subset!(bqp::BatchQuadraticModel{T, MT, VT, <:AbstractSparseOperator}, bx::AbstractMatrix, bc::AbstractMatrix, ::AbstractVector{<:Integer}) where {T, MT, VT} =
  NLPModels.cons!(bqp, bx, bc)

function cons_subset!(bqp::BatchQuadraticModel{T, MT, VT, <:BatchSparseOperator}, bx::AbstractMatrix, bc::AbstractMatrix, roots::AbstractVector{<:Integer}) where {T, MT, VT}
  batch_spmv_subset!(bc, bqp.A, bx, roots)
  return bc
end

# ---------- jac_coord_subset! / hess_coord_subset! ----------

"""
    jac_coord_subset!(bqp, bx, bjvals, roots)

Batched Jacobian values. Shared-A path broadcasts the single nzvals vector
across all columns; per-instance path gathers `A.nzvals[:, roots]`.
"""
jac_coord_subset!(bqp::BatchQuadraticModel{T, MT, VT, <:AbstractSparseOperator}, bx::AbstractMatrix, bjvals::AbstractMatrix, ::AbstractVector{<:Integer}) where {T, MT, VT} =
  NLPModels.jac_coord!(bqp, bx, bjvals)

function jac_coord_subset!(bqp::BatchQuadraticModel{T, MT, VT, <:BatchSparseOperator}, bx::AbstractMatrix, bjvals::AbstractMatrix, roots::AbstractVector{<:Integer}) where {T, MT, VT}
  gather_columns!(bjvals, bqp.A.nzvals, roots)
  return bjvals
end

"""
    hess_coord_subset!(bqp, bx, by, bobj_weight, bhvals, roots)

Batched Hessian values scaled by per-instance `bobj_weight`. Shared-Q path
uses the full-batch weighted outer product; per-instance path gathers
`Q.nzvals[:, roots]` then applies the weights.
"""
hess_coord_subset!(bqp::BatchQuadraticModel{T, MT, VT, AOp, <:AbstractSparseOperator}, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bhvals::AbstractMatrix, ::AbstractVector{<:Integer}) where {T, MT, VT, AOp} =
  NLPModels.hess_coord!(bqp, bx, by, bobj_weight, bhvals)

function hess_coord_subset!(bqp::BatchQuadraticModel{T, MT, VT, AOp, <:BatchSparseOperator}, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bhvals::AbstractMatrix, roots::AbstractVector{<:Integer}) where {T, MT, VT, AOp}
  gather_columns!(bhvals, bqp.Q.nzvals, roots)
  bhvals .*= bobj_weight'
  return bhvals
end
