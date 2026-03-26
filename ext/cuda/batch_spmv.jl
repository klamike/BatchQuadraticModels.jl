function _batch_spmv_impl!(
  out::AbstractMatrix{T}, op::BatchSparseOp{<:CuVector}, B::AbstractMatrix{T},
  alpha::T, beta::T, val_offset::Int32 = Int32(0),
) where {T}
  nout = Int32(length(op.rowptr) - 1)
  bs = Int32(size(out, 2))
  (nout == 0 || bs == 0) && return out
  if op.mean_row_nnz <= WARP_KERNEL_THRESHOLD
    _launch_scalar_kernel!(out, op, B, alpha, beta, val_offset, nout, bs)
  else
    _launch_warp_kernel!(out, op, B, alpha, beta, val_offset, nout, bs)
  end
  return out
end

_scalar_spmv_kernel!(
  out, A, B,
  flat_packed, rowptr,
  alpha, beta, val_offset::Int32, nout::Int32, bs::Int32,
) = begin
  j = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
  r = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y)

  (r > nout || j > bs) && return nothing

  acc = zero(eltype(out))
  @inbounds begin
    for k in rowptr[r]:(rowptr[r + Int32(1)] - Int32(1))
      packed = flat_packed[k]
      nz = Int32(packed >> 32)
      val = Int32(packed & 0xffffffff)
      acc += A[nz, j] * B[val + val_offset, j]
    end
    out[r, j] = iszero(beta) ? alpha * acc : alpha * acc + beta * out[r, j]
  end
  return nothing
end

function _launch_scalar_kernel!(
  out::AbstractMatrix{T}, op, B, alpha::T, beta::T,
  val_offset::Int32, nout::Int32, bs::Int32,
) where {T}
  tx, ty = Int32(32), Int32(8)
  threads = (tx, ty)
  blocks = (cld(Int(bs), Int(tx)), cld(Int(nout), Int(ty)))
  CUDA.@cuda always_inline = true threads = threads blocks = blocks _scalar_spmv_kernel!(
    out, op.nzVals, B, op.flat_packed, op.rowptr,
    alpha, beta, val_offset, nout, bs,
  )
end

_warp_spmv_kernel!(
  out, A, B,
  flat_packed, rowptr,
  alpha, beta, val_offset::Int32, nout::Int32, bs::Int32,
) = begin
  lane = Int32(threadIdx().x - Int32(1))
  r = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y)
  j = Int32(blockIdx().x)

  (r > nout || j > bs) && return nothing

  acc = zero(eltype(out))
  @inbounds begin
    start = rowptr[r]
    stop = rowptr[r + Int32(1)] - Int32(1)
    k = start + lane
    while k <= stop
      packed = flat_packed[k]
      nz = Int32(packed >> 32)
      val = Int32(packed & 0xffffffff)
      acc += A[nz, j] * B[val + val_offset, j]
      k += Int32(32)
    end
  end

  offset = Int32(16)
  while offset > Int32(0)
    acc += CUDA.shfl_down_sync(0xffffffff, acc, offset)
    offset >>= Int32(1)
  end

  @inbounds if lane == Int32(0)
    out[r, j] = iszero(beta) ? alpha * acc : alpha * acc + beta * out[r, j]
  end
  return nothing
end

function _launch_warp_kernel!(
  out::AbstractMatrix{T}, op, B, alpha::T, beta::T,
  val_offset::Int32, nout::Int32, bs::Int32,
) where {T}
  rows_per_block = Int32(4)
  threads = (Int32(32), rows_per_block)
  blocks = (Int(bs), cld(Int(nout), Int(rows_per_block)))
  CUDA.@cuda always_inline = true threads = threads blocks = blocks _warp_spmv_kernel!(
    out, op.nzVals, B, op.flat_packed, op.rowptr,
    alpha, beta, val_offset, nout, bs,
  )
end

function Adapt.adapt_structure(::Type{<:CuArray}, op::BatchSparseOp)
  BatchSparseOp(
    Adapt.adapt(CuArray, op.nzVals),
    Adapt.adapt(CuArray, op.rowptr),
    Adapt.adapt(CuArray, op.flat_nz),
    Adapt.adapt(CuArray, op.flat_val),
    Adapt.adapt(CuArray, op.flat_packed),
    op.mean_row_nnz,
  )
end
