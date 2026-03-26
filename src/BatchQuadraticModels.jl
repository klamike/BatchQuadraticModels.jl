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

export ObjRHSBatchQuadraticModel, BatchQuadraticModel
export BatchSparseOp, batch_spmv!, _batch_spmv_impl!, _build_op

include("batch_spmv.jl")
include("objrhsbatchqp.jl")
include("batchqp.jl")

end # module BatchQuadraticModels
