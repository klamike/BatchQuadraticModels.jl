module BatchQuadraticModelsKernelIntrinsicsExt

using KernelAbstractions
using KernelIntrinsics
import BatchQuadraticModels: BatchSparseOp

const BATCHSPMV_WARP_THRESHOLD = Ref(Int32(4))

function _batch_spmv_ki_impl!(
  out::AbstractMatrix{T}, op::BatchSparseOp, B::AbstractMatrix{T},
  alpha::T, beta::T, val_offset::Int32 = Int32(0),
) where {T}
  backend = KernelAbstractions.get_backend(out)
  KernelAbstractions.isgpu(backend) || return nothing

  nout = Int32(length(op.rowptr) - 1)
  bs = Int32(size(out, 2))
  (nout == 0 || bs == 0) && return out

  if op.mean_row_nnz <= BATCHSPMV_WARP_THRESHOLD[]
    _launch_scalar_kernel!(backend, out, op, B, alpha, beta, val_offset, nout, bs)
  else
    _launch_warp_kernel!(backend, out, op, B, alpha, beta, val_offset, nout, bs)
  end
  return out
end

@kernel function _scalar_spmv_kernel!(
  out, A, B,
  flat_packed, rowptr,
  alpha, beta, val_offset::Int32, nout::Int32, bs::Int32,
)
  j, r = @index(Global, NTuple)
  j = Int32(j)
  r = Int32(r)

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
end

function _launch_scalar_kernel!(
  backend, out::AbstractMatrix{T}, op, B, alpha::T, beta::T,
  val_offset::Int32, nout::Int32, bs::Int32,
) where {T}
  tx, ty = 32, 8
  _scalar_spmv_kernel!(backend)(
    out, op.nzVals, B, op.flat_packed, op.rowptr,
    alpha, beta, val_offset, nout, bs;
    ndrange = (Int(bs), Int(nout)),
    workgroupsize = (tx, ty),
  )
end

@kernel function _warp_spmv_kernel!(
  out, A, B,
  flat_packed, rowptr,
  alpha, beta, val_offset::Int32, nout::Int32, bs::Int32, warp_size::Int32,
)
  gx, r = @index(Global, NTuple)
  lane = Int32(@laneid() - 1)
  j = Int32(((gx - 1) ÷ warp_size) + 1)
  r = Int32(r)

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
      k += warp_size
    end
  end

  @warpfold(acc, +)

  @inbounds if lane == Int32(0)
    out[r, j] = iszero(beta) ? alpha * acc : alpha * acc + beta * out[r, j]
  end
end

function _launch_warp_kernel!(
  backend, out::AbstractMatrix{T}, op, B, alpha::T, beta::T,
  val_offset::Int32, nout::Int32, bs::Int32,
) where {T}
  warp_size = Int32(KernelIntrinsics.get_warpsize(out))
  rows_per_block = 4
  _warp_spmv_kernel!(backend)(
    out, op.nzVals, B, op.flat_packed, op.rowptr,
    alpha, beta, val_offset, nout, bs, warp_size;
    ndrange = (Int(bs) * Int(warp_size), Int(nout)),
    workgroupsize = (Int(warp_size), rows_per_block),
  )
end

end # module BatchQuadraticModelsKernelIntrinsicsExt
