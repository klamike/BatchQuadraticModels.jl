module BatchQuadraticModelsCUDAExt

using Adapt
using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
import LinearAlgebra
import LinearAlgebra: BlasFloat, Symmetric, mul!
using NLPModels
using SparseArrays
import BatchQuadraticModels
import BatchQuadraticModels:
  AbstractSparseOperator,
  _batch_spmv_impl!,
  _batch_spmv_subset_impl!,
  _adapt_to_operator,
  BatchSparseOp,
  BatchLinearModel,
  BatchQuadraticModel,
  ObjRHSLinearModel,
  ObjRHSBatchQuadraticModel,
  batch_mapreduce!,
  gather_columns!,
  gather_entries!,
  sparse_operator,
  operator_sparse_matrix
import QuadraticModels: QPData, SparseMatrixCOO, fill_structure!

const WARP_KERNEL_THRESHOLD = Int32(4)

include("cuda/operators.jl")
include("cuda/mapreduce.jl")
include("cuda/batch_spmv.jl")
include("cuda/obj_rhs.jl")
include("cuda/uniform.jl")

end # module BatchQuadraticModelsCUDAExt
