"""
    DeviceBatchSparseOperator

GPU batch operator. Layout mirrors [`HostBatchSparseOperator`](@ref) but
`rowptr` uses `Int32` and `(nz_idx, val_idx)` are packed into `Int64` for
coalesced kernel reads. `mean_row_nnz` selects scalar vs warp kernel per launch.
"""
struct DeviceBatchSparseOperator{MT,
                                 VI <: AbstractVector{Int},
                                 VI32 <: AbstractVector{Int32},
                                 VI64 <: AbstractVector{Int64}} <: BatchSparseOperator
  nzvals::MT
  rows::VI
  cols::VI
  rowptr::VI32
  packed::VI64
  mean_row_nnz::Float64
end

@inline _pack_nz_val(nz::Int32, val::Int32) = (Int64(nz) << 32) | Int64(val)

function _row_stats(rowptr::AbstractVector)
  n = length(rowptr) - 1
  n == 0 ? 0.0 : (rowptr[end] - rowptr[1]) / n
end

function _row_stats(rowptr::AnyCuArray)
  n = length(rowptr) - 1
  n == 0 && return 0.0
  nnz = Float64(sum(@view rowptr[end:end]) - sum(@view rowptr[1:1]))
  return nnz / n
end


function _batch_spmv_impl!(out::AbstractMatrix{T}, op::DeviceBatchSparseOperator, B::AbstractMatrix{T},
                           alpha::T, beta::T, val_offset::Int32 = Int32(0)) where {T}
  nout = Int32(length(op.rowptr) - 1); bs = Int32(size(out, 2))
  (nout == 0 || bs == 0) && return out
  if op.mean_row_nnz <= WARP_KERNEL_THRESHOLD
    _launch_scalar_kernel!(out, op, B, alpha, beta, val_offset, nout, bs)
  else
    _launch_warp_kernel!(out, op, B, alpha, beta, val_offset, nout, bs)
  end
  return out
end

function _batch_spmv_subset_impl!(out::AbstractMatrix{T}, op::DeviceBatchSparseOperator, B::AbstractMatrix{T},
                                  roots::AbstractVector{<:Integer},
                                  alpha::T, beta::T, val_offset::Int32 = Int32(0)) where {T}
  nout = Int32(length(op.rowptr) - 1); bs = Int32(length(roots))
  (nout == 0 || bs == 0) && return out
  if op.mean_row_nnz <= WARP_KERNEL_THRESHOLD
    _launch_scalar_subset_kernel!(out, op, B, roots, alpha, beta, val_offset, nout, bs)
  else
    _launch_warp_subset_kernel!(out, op, B, roots, alpha, beta, val_offset, nout, bs)
  end
  return out
end


_scalar_spmv_kernel!(out, A, B, flat_packed, rowptr, alpha, beta, val_offset::Int32, nout::Int32, bs::Int32) = begin
  j = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
  r = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y)
  (r > nout || j > bs) && return nothing
  acc = zero(eltype(out))
  @inbounds begin
    for k in rowptr[r]:(rowptr[r + Int32(1)] - Int32(1))
      packed = flat_packed[k]
      nz = Int32(packed >> 32); val = Int32(packed & 0xffffffff)
      acc += A[nz, j] * B[val + val_offset, j]
    end
    out[r, j] = iszero(beta) ? alpha * acc : alpha * acc + beta * out[r, j]
  end
  return nothing
end

function _launch_scalar_kernel!(out::AbstractMatrix{T}, op, B, alpha::T, beta::T,
                                val_offset::Int32, nout::Int32, bs::Int32) where {T}
  tx, ty = Int32(32), Int32(8)
  blocks = (cld(Int(bs), Int(tx)), cld(Int(nout), Int(ty)))
  CUDA.@cuda always_inline = true threads = (tx, ty) blocks = blocks _scalar_spmv_kernel!(
    out, op.nzvals, B, op.packed, op.rowptr, alpha, beta, val_offset, nout, bs)
end

_scalar_spmv_subset_kernel!(out, A, B, roots, flat_packed, rowptr,
                            alpha, beta, val_offset::Int32, nout::Int32, bs::Int32) = begin
  j = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
  r = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y)
  (r > nout || j > bs) && return nothing
  root_j = Int32(roots[j]); acc = zero(eltype(out))
  @inbounds begin
    for k in rowptr[r]:(rowptr[r + Int32(1)] - Int32(1))
      packed = flat_packed[k]
      nz = Int32(packed >> 32); val = Int32(packed & 0xffffffff)
      acc += A[nz, root_j] * B[val + val_offset, j]
    end
    out[r, j] = iszero(beta) ? alpha * acc : alpha * acc + beta * out[r, j]
  end
  return nothing
end

function _launch_scalar_subset_kernel!(out::AbstractMatrix{T}, op, B, roots, alpha::T, beta::T,
                                       val_offset::Int32, nout::Int32, bs::Int32) where {T}
  tx, ty = Int32(32), Int32(8)
  blocks = (cld(Int(bs), Int(tx)), cld(Int(nout), Int(ty)))
  CUDA.@cuda always_inline = true threads = (tx, ty) blocks = blocks _scalar_spmv_subset_kernel!(
    out, op.nzvals, B, roots, op.packed, op.rowptr, alpha, beta, val_offset, nout, bs)
end


_warp_spmv_kernel!(out, A, B, flat_packed, rowptr,
                   alpha, beta, val_offset::Int32, nout::Int32, bs::Int32) = begin
  lane = Int32(threadIdx().x - Int32(1))
  r = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y)
  j = Int32(blockIdx().x)
  (r > nout || j > bs) && return nothing
  acc = zero(eltype(out))
  @inbounds begin
    start = rowptr[r]; stop = rowptr[r + Int32(1)] - Int32(1)
    k = start + lane
    while k <= stop
      packed = flat_packed[k]
      nz = Int32(packed >> 32); val = Int32(packed & 0xffffffff)
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

function _launch_warp_kernel!(out::AbstractMatrix{T}, op, B, alpha::T, beta::T,
                              val_offset::Int32, nout::Int32, bs::Int32) where {T}
  rows_per_block = Int32(4)
  threads = (Int32(32), rows_per_block)
  blocks = (Int(bs), cld(Int(nout), Int(rows_per_block)))
  CUDA.@cuda always_inline = true threads = threads blocks = blocks _warp_spmv_kernel!(
    out, op.nzvals, B, op.packed, op.rowptr, alpha, beta, val_offset, nout, bs)
end

_warp_spmv_subset_kernel!(out, A, B, roots, flat_packed, rowptr,
                          alpha, beta, val_offset::Int32, nout::Int32, bs::Int32) = begin
  lane = Int32(threadIdx().x - Int32(1))
  r = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y)
  j = Int32(blockIdx().x)
  (r > nout || j > bs) && return nothing
  root_j = Int32(roots[j]); acc = zero(eltype(out))
  @inbounds begin
    start = rowptr[r]; stop = rowptr[r + Int32(1)] - Int32(1)
    k = start + lane
    while k <= stop
      packed = flat_packed[k]
      nz = Int32(packed >> 32); val = Int32(packed & 0xffffffff)
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

function _launch_warp_subset_kernel!(out::AbstractMatrix{T}, op, B, roots, alpha::T, beta::T,
                                     val_offset::Int32, nout::Int32, bs::Int32) where {T}
  rows_per_block = Int32(4)
  threads = (Int32(32), rows_per_block)
  blocks = (Int(bs), cld(Int(nout), Int(rows_per_block)))
  CUDA.@cuda always_inline = true threads = threads blocks = blocks _warp_spmv_subset_kernel!(
    out, op.nzvals, B, roots, op.packed, op.rowptr, alpha, beta, val_offset, nout, bs)
end


function Adapt.adapt_structure(::Type{<:CuArray}, op::HostBatchSparseOperator)
  rowptr  = Int32.(op.rowptr)
  nz_idx  = Int32.(op.nz_idx)
  val_idx = Int32.(op.val_idx)
  packed  = Vector{Int64}(undef, length(nz_idx))
  @inbounds for i in eachindex(packed)
    packed[i] = _pack_nz_val(nz_idx[i], val_idx[i])
  end
  return DeviceBatchSparseOperator(
    Adapt.adapt(CuArray, op.nzvals),
    Adapt.adapt(CuArray, op.rows),
    Adapt.adapt(CuArray, op.cols),
    Adapt.adapt(CuArray, rowptr),
    Adapt.adapt(CuArray, packed),
    _row_stats(rowptr),
  )
end

# CPU-built indices + GPU nzvals: build host op then adapt to device.
_build_op(nzvals::CuMatrix, rows, cols, rowptr, nz_map, val_map, colidx) =
  Adapt.adapt(CuArray, HostBatchSparseOperator(nzvals, rows, cols, rowptr, nz_map[colidx], val_map[colidx]))

# All-on-device build path: emits packed kernel directly.
function _build_op(nzvals::CuMatrix, rows::CuVector, cols::CuVector,
                   rowptr::CuVector, nz_map::CuVector, val_map::CuVector, colidx::CuVector)
  ncol = length(colidx)
  rowptr32 = Int32.(rowptr)
  packed = CUDA.zeros(Int64, ncol)
  if ncol > 0
    threads = 256; blocks = cld(ncol, threads)
    CUDA.@cuda threads=threads blocks=blocks _gather_pack_kernel!(packed, nz_map, val_map, colidx, Int32(ncol))
  end
  return DeviceBatchSparseOperator(nzvals, rows, cols, rowptr32, packed, _row_stats(rowptr))
end

function _gather_pack_kernel!(packed, nz_map, val_map, colidx, ncol::Int32)
  i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
  i > ncol && return nothing
  k = colidx[i]
  nz = Int32(nz_map[k]); vl = Int32(val_map[k])
  packed[i] = (Int64(nz) << 32) | Int64(vl)
  return nothing
end

# Device-side `_coo_to_csr`: sortperm + segmented count.
function _coo_to_csr(indices::CuVector{Int}, n::Int)
  nnz = length(indices)
  rowptr = CUDA.zeros(Int, n + 1)
  if nnz == 0
    fill!(view(rowptr, 1:1), 1)
    return rowptr, similar(indices, 0)
  end
  perm = CUDA.sortperm(indices)
  sorted = indices[perm]
  counts = CUDA.zeros(Int, n)
  CUDA.@cuda threads=256 blocks=cld(nnz, 256) _coo_count_kernel!(counts, sorted, Int32(nnz))
  cum = accumulate(+, counts)
  copyto!(view(rowptr, 2:n+1), cum)
  rowptr .+= 1
  return rowptr, perm
end

function _coo_count_kernel!(counts, sorted, nnz::Int32)
  i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
  i > nnz && return nothing
  CUDA.@atomic counts[sorted[i]] += 1
  return nothing
end
