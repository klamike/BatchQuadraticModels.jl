"""
    _coo_to_csr(indices, n) -> (rowptr, colidx)

Convert COO row indices to CSR row pointers together with the permutation that
groups nonzeros by row.
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

abstract type BatchSparseOp end

struct HostBatchSparseOp{MT, VI <: AbstractVector{Int}} <: BatchSparseOp
  nzvals::MT
  rowptr::VI
  nz_idx::VI
  val_idx::VI
end

struct DeviceBatchSparseOp{MT, VI32 <: AbstractVector{Int32}, VI64 <: AbstractVector{Int64}} <: BatchSparseOp
  nzvals::MT
  rowptr::VI32
  packed::VI64
  mean_row_nnz::Float64
end

@inline _pack_nz_val(nz::Int32, val::Int32) = (Int64(nz) << 32) | Int64(val)

function _row_stats(rowptr::AbstractVector)
  nrows = length(rowptr) - 1
  nrows == 0 && return 0.0
  total = 0
  @inbounds for r in 1:nrows
    total += rowptr[r + 1] - rowptr[r]
  end
  return total / nrows
end

function _build_host_op(nzvals, rowptr, nz_map, val_map, colidx)
  nz_idx = similar(nz_map, Int, length(colidx))
  val_idx = similar(val_map, Int, length(colidx))
  if !isempty(colidx)
    nz_idx .= nz_map[colidx]
    val_idx .= val_map[colidx]
  end
  return HostBatchSparseOp(nzvals, rowptr, nz_idx, val_idx)
end

_build_op(nzvals::Matrix, rowptr, nz_map, val_map, colidx) = _build_host_op(nzvals, rowptr, nz_map, val_map, colidx)

function batch_spmv!(
  out::AbstractMatrix{T}, op::BatchSparseOp, B::AbstractMatrix,
  alpha::T = one(T), beta::T = zero(T); val_offset::Int = 0,
) where {T}
  _batch_spmv_impl!(out, op, B, alpha, beta, Int32(val_offset))
end

function batch_spmv_subset!(
  out::AbstractMatrix{T},
  op::BatchSparseOp,
  B::AbstractMatrix,
  roots::AbstractVector{<:Integer},
  alpha::T = one(T),
  beta::T = zero(T);
  val_offset::Int = 0,
) where {T}
  _batch_spmv_subset_impl!(out, op, B, roots, alpha, beta, Int32(val_offset))
end

function _batch_spmv_impl!(
  out::AbstractMatrix{T}, op::HostBatchSparseOp, B::AbstractMatrix,
  alpha::T, beta::T, val_offset::Int32 = Int32(0),
) where {T}
  nout = length(op.rowptr) - 1
  bs = size(out, 2)
  beta_is_zero = iszero(beta)
  @inbounds for r in 1:nout
    for j in 1:bs
      acc = zero(T)
      for k in op.rowptr[r]:(op.rowptr[r + 1] - 1)
        acc += op.nzvals[op.nz_idx[k], j] * B[op.val_idx[k] + val_offset, j]
      end
      out[r, j] = beta_is_zero ? alpha * acc : alpha * acc + beta * out[r, j]
    end
  end
  return out
end

function _batch_spmv_subset_impl!(
  out::AbstractMatrix{T},
  op::HostBatchSparseOp,
  B::AbstractMatrix,
  roots::AbstractVector{<:Integer},
  alpha::T,
  beta::T,
  val_offset::Int32 = Int32(0),
) where {T}
  nout = length(op.rowptr) - 1
  bs = length(roots)
  beta_is_zero = iszero(beta)
  @inbounds for r in 1:nout
    for j in 1:bs
      root_j = Int(roots[j])
      acc = zero(T)
      for k in op.rowptr[r]:(op.rowptr[r + 1] - 1)
        acc += op.nzvals[op.nz_idx[k], root_j] * B[op.val_idx[k] + val_offset, j]
      end
      out[r, j] = beta_is_zero ? alpha * acc : alpha * acc + beta * out[r, j]
    end
  end
  return out
end
