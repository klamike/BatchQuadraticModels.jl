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

function gpu_operator end
function operator_sparse_matrix end
function batch_mapreduce! end
function batch_maximum! end
function batch_minimum! end
function batch_sum! end

export ObjRHSBatchQuadraticModel, BatchQuadraticModel
export BatchSparseOp, batch_spmv!, _batch_spmv_impl!, _build_op
export batch_mapreduce!, batch_maximum!, batch_minimum!, batch_sum!

include("batch_mapreduce.jl")
include("batch_spmv.jl")
include("models/uniform.jl")
include("models/obj_rhs.jl")

end # module BatchQuadraticModels
