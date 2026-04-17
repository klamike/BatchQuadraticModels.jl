function _batch_spmv_impl!(
  out::AbstractMatrix{T}, op::DeviceBatchSparseOp, B::AbstractMatrix{T},
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

function BatchQuadraticModels._batch_spmv_subset_impl!(
  out::AbstractMatrix{T},
  op::DeviceBatchSparseOp,
  B::AbstractMatrix{T},
  roots::AbstractVector{<:Integer},
  alpha::T,
  beta::T,
  val_offset::Int32 = Int32(0),
) where {T}
  nout = Int32(length(op.rowptr) - 1)
  bs = Int32(length(roots))
  (nout == 0 || bs == 0) && return out
  if op.mean_row_nnz <= WARP_KERNEL_THRESHOLD
    _launch_scalar_subset_kernel!(out, op, B, roots, alpha, beta, val_offset, nout, bs)
  else
    _launch_warp_subset_kernel!(out, op, B, roots, alpha, beta, val_offset, nout, bs)
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
    out, op.nzvals, B, op.packed, op.rowptr,
    alpha, beta, val_offset, nout, bs,
  )
end

_scalar_spmv_subset_kernel!(
  out, A, B, roots,
  flat_packed, rowptr,
  alpha, beta, val_offset::Int32, nout::Int32, bs::Int32,
) = begin
  j = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
  r = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y)

  (r > nout || j > bs) && return nothing

  root_j = Int32(roots[j])
  acc = zero(eltype(out))
  @inbounds begin
    for k in rowptr[r]:(rowptr[r + Int32(1)] - Int32(1))
      packed = flat_packed[k]
      nz = Int32(packed >> 32)
      val = Int32(packed & 0xffffffff)
      acc += A[nz, root_j] * B[val + val_offset, j]
    end
    out[r, j] = iszero(beta) ? alpha * acc : alpha * acc + beta * out[r, j]
  end
  return nothing
end

function _launch_scalar_subset_kernel!(
  out::AbstractMatrix{T}, op, B, roots, alpha::T, beta::T,
  val_offset::Int32, nout::Int32, bs::Int32,
) where {T}
  tx, ty = Int32(32), Int32(8)
  threads = (tx, ty)
  blocks = (cld(Int(bs), Int(tx)), cld(Int(nout), Int(ty)))
  CUDA.@cuda always_inline = true threads = threads blocks = blocks _scalar_spmv_subset_kernel!(
    out, op.nzvals, B, roots, op.packed, op.rowptr,
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
    out, op.nzvals, B, op.packed, op.rowptr,
    alpha, beta, val_offset, nout, bs,
  )
end

_warp_spmv_subset_kernel!(
  out, A, B, roots,
  flat_packed, rowptr,
  alpha, beta, val_offset::Int32, nout::Int32, bs::Int32,
) = begin
  lane = Int32(threadIdx().x - Int32(1))
  r = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y)
  j = Int32(blockIdx().x)

  (r > nout || j > bs) && return nothing

  root_j = Int32(roots[j])
  acc = zero(eltype(out))
  @inbounds begin
    start = rowptr[r]
    stop = rowptr[r + Int32(1)] - Int32(1)
    k = start + lane
    while k <= stop
      packed = flat_packed[k]
      nz = Int32(packed >> 32)
      val = Int32(packed & 0xffffffff)
      acc += A[nz, root_j] * B[val + val_offset, j]
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

function _launch_warp_subset_kernel!(
  out::AbstractMatrix{T}, op, B, roots, alpha::T, beta::T,
  val_offset::Int32, nout::Int32, bs::Int32,
) where {T}
  rows_per_block = Int32(4)
  threads = (Int32(32), rows_per_block)
  blocks = (Int(bs), cld(Int(nout), Int(rows_per_block)))
  CUDA.@cuda always_inline = true threads = threads blocks = blocks _warp_spmv_subset_kernel!(
    out, op.nzvals, B, roots, op.packed, op.rowptr,
    alpha, beta, val_offset, nout, bs,
  )
end

function Adapt.adapt_structure(::Type{<:CuArray}, op::HostBatchSparseOp)
  rowptr = Int32.(op.rowptr)
  nz_idx = Int32.(op.nz_idx)
  val_idx = Int32.(op.val_idx)
  packed = Vector{Int64}(undef, length(nz_idx))
  @inbounds for i in eachindex(packed)
    packed[i] = _pack_nz_val(nz_idx[i], val_idx[i])
  end
  return DeviceBatchSparseOp(
    Adapt.adapt(CuArray, op.nzvals),
    Adapt.adapt(CuArray, rowptr),
    Adapt.adapt(CuArray, packed),
    _row_stats(rowptr),
  )
end

function BatchQuadraticModels._build_op(nzvals::CuMatrix, rowptr, nz_map, val_map, colidx)
  return Adapt.adapt(CuArray, BatchQuadraticModels._build_host_op(nzvals, rowptr, nz_map, val_map, colidx))
end
