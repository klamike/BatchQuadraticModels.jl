module BatchQuadraticModelsCUDAExt

using Adapt
using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
import LinearAlgebra
import LinearAlgebra: BlasFloat, Symmetric, Transpose, mul!
using NLPModels
using SparseArrays
import BatchQuadraticModels
import BatchQuadraticModels:
  AbstractSparseOperator,
  _mul_jt!,
  _batch_spmv_impl!,
  _batch_spmv_subset_impl!,
  BatchSparseOp,
  DeviceBatchSparseOp,
  HostBatchSparseOp,
  BatchLinearModel,
  BatchQuadraticModel,
  LinearModel,
  LPData,
  QPData,
  QuadraticModel,
  ObjRHSBatchLinearModel,
  ObjRHSBatchQuadraticModel,
  _copy_sparse_structure!,
  _copy_sparse_values!,
  batch_mapreduce!,
  gather_columns!,
  gather_entries!,
  sparse_operator,
  operator_sparse_matrix
import SparseMatricesCOO: SparseMatrixCOO

const WARP_KERNEL_THRESHOLD = Int32(4)

include("cuda/operators.jl")
include("cuda/single.jl")
include("cuda/mapreduce.jl")
include("cuda/batch_spmv.jl")
include("cuda/obj_rhs.jl")
include("cuda/uniform.jl")

end # module BatchQuadraticModelsCUDAExt
