"""
    BatchSparseOperator

Abstract parent for operators carrying per-instance nzvals over a shared
sparsity pattern. CPU implementation: [`HostBatchSparseOperator`](@ref); GPU
implementation: `DeviceBatchSparseOperator` (CUDA extension).
"""
abstract type BatchSparseOperator end

"""
    HostBatchSparseOperator

CPU batch operator. `nzvals::MT` is `(nnz_expanded, nbatch)`. `rows`/`cols`
hold the original COO indices (used by structure queries); `rowptr`/`nz_idx`/
`val_idx` drive the per-row SpMV loop.
"""
struct HostBatchSparseOperator{MT, VI <: AbstractVector{Int}} <: BatchSparseOperator
  nzvals::MT
  rows::VI
  cols::VI
  rowptr::VI
  nz_idx::VI
  val_idx::VI
end

_sparse_structure(A::BatchSparseOperator) = (A.rows, A.cols)
_sparse_values(A::BatchSparseOperator)    = A.nzvals
_copy_sparse_structure!(A::BatchSparseOperator, rows::AbstractVector, cols::AbstractVector) =
  (copyto!(rows, A.rows); copyto!(cols, A.cols); (rows, cols))

"""
    _coo_to_csr(indices, n) -> (rowptr, colidx)

Convert COO row indices to CSR row pointers along with the permutation that
groups nonzeros by row: `colidx[rowptr[r]:rowptr[r+1]-1]` lists original COO
indices in row `r`.
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

# Symmetric Q expansion: each off-diagonal entry contributes twice (lower + upper).
function _symmetric_scatter_ops(rows::AbstractVector{Int}, cols::AbstractVector{Int}, nnz::Int)
  off_diag = findall(rows .!= cols)
  scatter_rows = vcat(rows, cols[off_diag])
  nz_idx       = vcat(collect(1:nnz), off_diag)
  gather_cols  = vcat(cols, rows[off_diag])
  return scatter_rows, nz_idx, gather_cols
end

# CUDA ext overrides for `nzvals::CuMatrix` (builds DeviceBatchSparseOperator).
_build_op(nzvals::Matrix, rows, cols, rowptr, nz_map, val_map, colidx) =
  HostBatchSparseOperator(nzvals, rows, cols, rowptr, nz_map[colidx], val_map[colidx])

# Per-instance Jacobian/Hessian builders used by the BQM constructors.
function _jacobian_op(qp_ref, nzvals)
  rows, cols = _sparse_structure(qp_ref.data.A)
  rowptr, colidx = _coo_to_csr(rows, qp_ref.meta.ncon)
  return _build_op(nzvals, rows, cols, rowptr, collect(1:qp_ref.meta.nnzj), cols, colidx)
end

function _hessian_op(qp_ref, nzvals)
  rows, cols = _sparse_structure(qp_ref.data.Q)
  sym_rows, sym_nz, sym_cols = _symmetric_scatter_ops(rows, cols, qp_ref.meta.nnzh)
  rowptr, colidx = _coo_to_csr(sym_rows, qp_ref.meta.nvar)
  return _build_op(nzvals, rows, cols, rowptr, sym_nz, sym_cols, colidx)
end


"""
    batch_spmv!(out, op, B[, alpha=1, beta=0]; val_offset=0)

Batched SpMV: `out = α op B + β out`, one column of `op.nzvals` per column of
`out`/`B`. `val_offset` shifts the row index used to look up into `B`.
"""
batch_spmv!(out::AbstractMatrix{T}, op::BatchSparseOperator, B::AbstractMatrix,
            alpha::T = one(T), beta::T = zero(T); val_offset::Int = 0) where {T} =
  _batch_spmv_impl!(out, op, B, alpha, beta, Int32(val_offset))

"""
    batch_spmv_subset!(out, op, B, roots[, alpha=1, beta=0]; val_offset=0)

Subset variant: output column `j` reads from `op.nzvals[:, roots[j]]`. Used by
the IPM to skip converged (inactive) batch instances.
"""
batch_spmv_subset!(out::AbstractMatrix{T}, op::BatchSparseOperator, B::AbstractMatrix,
                   roots::AbstractVector{<:Integer}, alpha::T = one(T), beta::T = zero(T);
                   val_offset::Int = 0) where {T} =
  _batch_spmv_subset_impl!(out, op, B, roots, alpha, beta, Int32(val_offset))

LinearAlgebra.mul!(Y::AbstractMatrix{T}, op::BatchSparseOperator, X::AbstractMatrix{T}, α::Number, β::Number) where {T} =
  batch_spmv!(Y, op, X, T(α), T(β))


@inline function _batch_spmv_core!(out::AbstractMatrix{T}, op::HostBatchSparseOperator, B,
                                   alpha::T, beta::T, val_offset::Int32, nzcol::F) where {T, F}
  nout = length(op.rowptr) - 1
  beta_zero = iszero(beta)
  @inbounds for r in 1:nout, j in 1:size(out, 2)
    jz = nzcol(j)
    acc = zero(T)
    for k in op.rowptr[r]:(op.rowptr[r + 1] - 1)
      acc += op.nzvals[op.nz_idx[k], jz] * B[op.val_idx[k] + val_offset, j]
    end
    out[r, j] = beta_zero ? alpha * acc : alpha * acc + beta * out[r, j]
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


"""
    batch_mapreduce!(f, op, neutral, out, srcs...)

Column-wise `mapreduce` over batch matrices: `out[1, j] = mapreduce(f, op, srcs[:, j]...; init=neutral)`.
GPU extension overrides for `out::CuMatrix`.
"""
batch_mapreduce!(f, op, neutral, out::AbstractMatrix, srcs::AbstractMatrix...) =
  (out .= mapreduce(f, op, srcs...; dims = 1, init = neutral))

batch_maximum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, max, typemin(T), out, src)
batch_minimum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, min, typemax(T), out, src)
batch_sum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
  batch_mapreduce!(identity, +, zero(T), out, src)

"""
    gather_columns!(dst, src, roots)

`dst[:, j] = src[:, roots[j]]` for `j in eachindex(roots)`.
"""
function gather_columns!(dst::AbstractMatrix, src::AbstractMatrix, roots::AbstractVector{<:Integer})
  @assert size(dst, 1) == size(src, 1)
  @assert size(dst, 2) >= length(roots)
  @inbounds for j in eachindex(roots)
    copyto!(view(dst, :, j), view(src, :, Int(roots[j])))
  end
  return dst
end

"""
    gather_entries!(dst, src, roots)

Vector counterpart: `dst[j] = src[roots[j]]`.
"""
function gather_entries!(dst::AbstractVector, src::AbstractVector, roots::AbstractVector{<:Integer})
  @assert length(dst) >= length(roots)
  @inbounds for j in eachindex(roots)
    dst[j] = src[Int(roots[j])]
  end
  return dst
end
