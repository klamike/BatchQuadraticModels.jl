"""
    _coo_to_csr(indices, n) -> (rowptr, colidx)

Convert COO row/column indices to CSR format.
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

struct BatchSparseOp{VI, VI64, MT}
  nzVals::MT
  rowptr::VI
  flat_nz::VI
  flat_val::VI
  flat_packed::VI64  # TODO: benchmark packed vs unpacked
  mean_row_nnz::Float64
end

function Adapt.adapt_structure(to, op::BatchSparseOp)
  return BatchSparseOp(
    Adapt.adapt(to, op.nzVals),
    Adapt.adapt(to, op.rowptr),
    Adapt.adapt(to, op.flat_nz),
    Adapt.adapt(to, op.flat_val),
    Adapt.adapt(to, op.flat_packed),
    op.mean_row_nnz,
  )
end

@inline _pack_nz_val(nz::Int32, val::Int32) = (Int64(nz) << 32) | Int64(val)
@inline _unpack_nz(packed::Int64) = Int32(packed >> 32)
@inline _unpack_val(packed::Int64) = Int32(packed & 0xffffffff)

# Compute row nnz stats so CUDA extension can decide when to use scalar vs warp kernels
function _row_stats(rowptr::AbstractVector)
  nrows = length(rowptr) - 1
  nrows == 0 && return Float64(0)
  total = Int32(0)
  @inbounds for r in 1:nrows
    rl = Int32(rowptr[r + 1] - rowptr[r])
    total += rl
  end
  return Float64(total / nrows)
end

function _build_op(nzVals, rowptr, nz_map, val_map, colidx)
  rowptr32 = Int32.(rowptr)
  mean_nnz = _row_stats(rowptr32)
  flat_nz = similar(nz_map, Int32, length(colidx))
  flat_val = similar(val_map, Int32, length(colidx))
  if length(colidx) > 0
    flat_nz .= nz_map[colidx]
    flat_val .= val_map[colidx]
  end
  flat_packed = Vector{Int64}(undef, length(colidx))
  for i in eachindex(flat_packed)
    flat_packed[i] = _pack_nz_val(flat_nz[i], flat_val[i])
  end
  return BatchSparseOp(nzVals, rowptr32, flat_nz, flat_val, flat_packed, mean_nnz)
end

function batch_spmv!(
  out::AbstractMatrix{T}, op::BatchSparseOp, B::AbstractMatrix,
  alpha::T = one(T), beta::T = zero(T); val_offset::Int = 0,
) where {T}
  _batch_spmv_impl!(out, op, B, alpha, beta, Int32(val_offset))
end

function _batch_spmv_impl!(
  out::AbstractMatrix{T}, op::BatchSparseOp, B::AbstractMatrix,
  alpha::T, beta::T, val_offset::Int32 = Int32(0),
) where {T}
  nout = length(op.rowptr) - 1
  bs = size(out, 2)
  beta_is_zero = iszero(beta)
  @inbounds for r in 1:nout
    for j in 1:bs
      acc = zero(T)
      for k in op.rowptr[r]:(op.rowptr[r + 1] - 1)
        acc += op.nzVals[op.flat_nz[k], j] * B[op.flat_val[k] + val_offset, j]
      end
      out[r, j] = beta_is_zero ? alpha * acc : alpha * acc + beta * out[r, j]
    end
  end
  return out
end
