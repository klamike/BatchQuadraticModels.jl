module BatchQuadraticModels

using Adapt
using LinearAlgebra, SparseArrays
using NLPModels
using SparseMatricesCOO

abstract type AbstractSparseOperator{T} <: AbstractMatrix{T} end

# Forward declaration for the MOI ext to extend (defined in `ext/moi/qp_model.jl`).
function qp_model end

include("batch_utils.jl")
include("batch_mapreduce.jl")
include("batch_spmv.jl")
include("wrapper.jl")

include("models/single.jl")
include("models/batch/entry.jl")
include("models/batch/models.jl")
include("models/batch/subset.jl")

include("standard_form/scalar.jl")
include("standard_form/scalar_build.jl")
include("standard_form/kernels.jl")
include("standard_form/batch.jl")

export LinearModel, QuadraticModel, LPData, QPData
export BatchQuadraticModel, ObjRHSBatchQuadraticModel, UniformBatchQuadraticModel, batch_model
export BatchSparseOperator, batch_spmv!, batch_spmv_subset!
export batch_mapreduce!, batch_maximum!, batch_minimum!, batch_sum!
export gather_columns!, gather_entries!
export obj_subset!, grad_subset!, cons_subset!, jac_coord_subset!, hess_coord_subset!
export standard_form, update_standard_form!
export recover_primal!, recover_primal, recover_variable_multipliers!
export StandardFormWorkspace, StandardFormBatchWorkspace

end # module BatchQuadraticModels
