module BatchQuadraticModelsCUDAExt

using Adapt
using Atomix: Atomix
using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
using LinearAlgebra
using NLPModels
using SparseArrays
import LinearAlgebra: BlasFloat, Symmetric, Transpose, mul!, tril
import SparseMatricesCOO: SparseMatrixCOO

import BatchQuadraticModels:
  AbstractSparseOperator, SparseOperator, BatchSparseOperator, HostBatchSparseOperator,
  LPData, QPData, LinearModel, QuadraticModel,
  BatchQuadraticModel,
  ScatterMap, BoundMap, BoundKind, BK_NONE,
  VAR_LB, VAR_LB_UB, VAR_UB, VAR_FREE, CON_LB, CON_RANGE, CON_UB, CON_EQ,
  StandardFormWorkspace,
  sparse_operator, operator_sparse_matrix,
  standard_form, update_standard_form!,
  _mul_jt!, _batch_spmv_impl!, _batch_spmv_subset_impl!,
  _copy_sparse_structure!, _copy_sparse_values!,
  _sparse_structure, _sparse_values,
  _apply_scatter_map!, _build_scalar_sparse,
  _recover_primal_apply!, _scatter_multipliers!, _gather_dual!,
  _update_x_offset!, _update_var_start!, _update_constraint_start!,
  _update_dual_start!, _update_rhs_base!, _apply_rhs_shift!, _coldot!, _add_dot!,
  _build_op, _coo_to_csr,
  _build_standard_layout, _build_c_map, _standard_var_width,
  _adapt_batch_meta, _adapt_to_batch_backend,
  batch_mapreduce!, gather_columns!, gather_entries!

const WARP_KERNEL_THRESHOLD = Int32(4)

include("sparse_operator.jl")
include("models.jl")
include("mapreduce.jl")
include("batch_spmv.jl")
include("batch_models.jl")
include("standard_form_kernels.jl")
include("standard_form.jl")

end # module
