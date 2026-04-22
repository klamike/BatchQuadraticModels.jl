# ---------- structural helpers ----------

"""
    _coo_to_csr(indices, n) -> (rowptr, colidx)

Convert COO row indices to CSR row pointers together with the permutation that
groups nonzeros by row. `colidx[rowptr[r]:rowptr[r+1]-1]` lists the original
COO indices of row `r`.
"""
function _coo_to_csr(indices::AbstractVector{Int}, n::Int)
  nnz = length(indices)
  rowptr = zeros(Int, n + 1)
  for i in 1:nnz
    rowptr[indices[i] + 1] += 1
  end
  rowptr[1] = 1
  for r in 1:n
    rowptr[r + 1] += rowptr[r]
  end
  colidx = Vector{Int}(undef, nnz)
  pos = copy(rowptr[1:n])
  for i in 1:nnz
    r = indices[i]
    colidx[pos[r]] = i
    pos[r] += 1
  end
  return rowptr, colidx
end

"""
    _symmetric_scatter_ops(rows, cols, nnz)

Given the lower-triangular `(rows, cols)` structure of a symmetric matrix with
`nnz` nonzeros, produce the scatter/gather triples for the expanded symmetric
operator: each off-diagonal source entry contributes twice (lower + upper).
Returns `(scatter_rows, nz_idx, gather_cols)` with lengths `nnz + n_offdiag`.
"""
function _symmetric_scatter_ops(rows::AbstractVector{Int}, cols::AbstractVector{Int}, nnz::Int)
  off_diag = findall(rows .!= cols)
  scatter_rows = vcat(rows, cols[off_diag])
  nz_idx       = vcat(_indices_like(rows, nnz), off_diag)
  gather_cols  = vcat(cols, rows[off_diag])
  return scatter_rows, nz_idx, gather_cols
end

# ---------- batch-sparse-operator types ----------

"""
    BatchSparseOperator

Abstract parent for operators that carry per-instance nzvals sharing a common
sparsity pattern. Subtypes plug into the `batch_spmv!` dispatch:
[`HostBatchSparseOperator`](@ref) on CPU, `DeviceBatchSparseOperator` on GPU.
"""
abstract type BatchSparseOperator end

"""
    HostBatchSparseOperator{MT, VI}

CPU batch operator. `nzvals::MT` is `(nnz_expanded, nbatch)` — each batch
column holds the per-instance nonzero values over the expanded (symmetric-
scattered) structure. `rows`/`cols` carry the *original* COO indices (used by
`jac_structure!`/`hess_structure!` and to materialize scalar representative
matrices). `rowptr`, `nz_idx`, `val_idx` drive the per-row SpMV loop.
"""
struct HostBatchSparseOperator{MT, VI <: AbstractVector{Int}} <: BatchSparseOperator
  nzvals::MT
  rows::VI
  cols::VI
  rowptr::VI
  nz_idx::VI
  val_idx::VI
end

"""
    DeviceBatchSparseOperator{MT, VI, VI32, VI64}

GPU batch operator. Layout mirrors [`HostBatchSparseOperator`](@ref) but
`rowptr` uses `Int32` and `(nz_idx, val_idx)` are packed into a single
`Int64` array `packed = (nz << 32) | val` for coalesced kernel reads.
`mean_row_nnz` picks the scalar- vs warp-kernel variant per launch.
"""
struct DeviceBatchSparseOperator{MT, VI <: AbstractVector{Int}, VI32 <: AbstractVector{Int32}, VI64 <: AbstractVector{Int64}} <: BatchSparseOperator
  nzvals::MT
  rows::VI
  cols::VI
  rowptr::VI32
  packed::VI64
  mean_row_nnz::Float64
end

# Pack a `(nz, val)` index pair into a single `Int64` — hot-loop GPU dispatch.
@inline _pack_nz_val(nz::Int32, val::Int32) = (Int64(nz) << 32) | Int64(val)

# Mean row density for the warp-vs-scalar GPU kernel heuristic.
function _row_stats(rowptr::AbstractVector)
  nrows = length(rowptr) - 1
  nrows == 0 ? 0.0 : (rowptr[end] - rowptr[1]) / nrows
end

# ---------- builders ----------

# Assemble a host batch operator by gathering `(nz_map, val_map)` along the
# row-major `colidx` permutation from `_coo_to_csr`.
function _build_host_op(nzvals, rows, cols, rowptr, nz_map, val_map, colidx)
  nz_idx  = nz_map[colidx]
  val_idx = val_map[colidx]
  return HostBatchSparseOperator(nzvals, rows, cols, rowptr, nz_idx, val_idx)
end

# `nzvals::Matrix` dispatch lands on the host variant; GPU ext overrides with
# `nzvals::AnyCuArray` to build a `DeviceBatchSparseOperator`.
_build_op(nzvals::Matrix, rows, cols, rowptr, nz_map, val_map, colidx) =
  _build_host_op(nzvals, rows, cols, rowptr, nz_map, val_map, colidx)

# ---------- public SpMV API ----------

"""
    batch_spmv!(out, op::BatchSparseOperator, B[, alpha=1, beta=0]; val_offset=0)

Batched sparse matrix-vector: `out = alpha * op * B + beta * out`, one column
of `op.nzvals` per column of `out`/`B`. `val_offset` shifts the row index used
to look up into `B` (for SpMV on a reduced coordinate system).
"""
function batch_spmv!(
  out::AbstractMatrix{T}, op::BatchSparseOperator, B::AbstractMatrix,
  alpha::T = one(T), beta::T = zero(T); val_offset::Int = 0,
) where {T}
  _batch_spmv_impl!(out, op, B, alpha, beta, Int32(val_offset))
end

"""
    batch_spmv_subset!(out, op, B, roots[, alpha=1, beta=0]; val_offset=0)

Subset variant of [`batch_spmv!`](@ref): output column `j` reads nzvals from
`op.nzvals[:, roots[j]]`. Skips inactive batch instances when converged
entries drop out of the IPM's active set.
"""
function batch_spmv_subset!(
  out::AbstractMatrix{T},
  op::BatchSparseOperator,
  B::AbstractMatrix,
  roots::AbstractVector{<:Integer},
  alpha::T = one(T),
  beta::T = zero(T);
  val_offset::Int = 0,
) where {T}
  _batch_spmv_subset_impl!(out, op, B, roots, alpha, beta, Int32(val_offset))
end

# `mul!` override so callers can treat a `BatchSparseOperator` like any other
# sparse operator. Dispatches into `batch_spmv!` for the actual work.
LinearAlgebra.mul!(Y::AbstractMatrix{T}, op::BatchSparseOperator, X::AbstractMatrix{T}, α::Number, β::Number) where {T} =
  batch_spmv!(Y, op, X, T(α), T(β))

# ---------- CPU SpMV implementation ----------

# `nzcol(j)` maps the output column to the nzvals column — identity for the
# full-batch path, `roots[j]` for the subset path. Keeps the kernel body
# single-source: both impls are the same accumulate-over-nonzeros loop.
@inline function _batch_spmv_core!(out::AbstractMatrix{T}, op::HostBatchSparseOperator, B,
                                    alpha::T, beta::T, val_offset::Int32, nzcol::F) where {T, F}
  nout = length(op.rowptr) - 1
  beta_is_zero = iszero(beta)
  @inbounds for r in 1:nout, j in 1:size(out, 2)
    jz = nzcol(j)
    acc = zero(T)
    for k in op.rowptr[r]:(op.rowptr[r + 1] - 1)
      acc += op.nzvals[op.nz_idx[k], jz] * B[op.val_idx[k] + val_offset, j]
    end
    out[r, j] = beta_is_zero ? alpha * acc : alpha * acc + beta * out[r, j]
  end
  return out
end

_batch_spmv_impl!(out::AbstractMatrix{T}, op::HostBatchSparseOperator, B::AbstractMatrix,
                   alpha::T, beta::T, val_offset::Int32 = Int32(0)) where {T} =
  _batch_spmv_core!(out, op, B, alpha, beta, val_offset, identity)

_batch_spmv_subset_impl!(out::AbstractMatrix{T}, op::HostBatchSparseOperator, B::AbstractMatrix,
                          roots::AbstractVector{<:Integer}, alpha::T, beta::T,
                          val_offset::Int32 = Int32(0)) where {T} =
  _batch_spmv_core!(out, op, B, alpha, beta, val_offset, j -> Int(@inbounds roots[j]))
