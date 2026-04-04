module BatchQuadraticModels

using Adapt
using LinearAlgebra, SparseArrays
using NLPModels
using QuadraticModels
using SparseMatricesCOO

import QuadraticModels:
  QPData,
  QuadraticModel,
  fill_structure!

abstract type AbstractSparseOperator{T} <: AbstractMatrix{T} end
abstract type AbstractBatchQuadraticModel{T, MT} <: NLPModels.AbstractBatchNLPModel{T, MT} end
abstract type AbstractUniformBatchQuadraticModel{T, MT} <: AbstractBatchQuadraticModel{T, MT} end
abstract type AbstractObjRHSBatchQuadraticModel{T, MT} <: AbstractBatchQuadraticModel{T, MT} end

function sparse_operator end
function operator_sparse_matrix end

export ObjRHSBatchQuadraticModel, BatchQuadraticModel
export ObjRHSLinearModel, BatchLinearModel, batch_model
export BatchSparseOp, batch_spmv!, _batch_spmv_impl!, _build_op
export batch_mapreduce!, batch_maximum!, batch_minimum!, batch_sum!
export gather_columns!, gather_entries!, batch_spmv_subset!
export obj_subset!, grad_subset!, cons_subset!, jac_coord_subset!, hess_coord_subset!

include("batch_mapreduce.jl")
include("batch_spmv.jl")
include("operators.jl")
include("models/uniform.jl")
include("models/obj_rhs.jl")
include("models/linear.jl")

end # module BatchQuadraticModels
