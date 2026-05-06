module BatchQuadraticModels

using Adapt
using LinearAlgebra, SparseArrays
using NLPModels
using SparseMatricesCOO

abstract type AbstractSparseOperator{T} <: AbstractMatrix{T} end

# Forward declaration for the MOI ext (defined in `ext/moi/qp_model.jl`).
function qp_model end

include("utils.jl")
include("sparse_operator.jl")
include("models.jl")
include("batch_spmv.jl")
include("batch_models.jl")
include("standard_form_types.jl")
include("standard_form_kernels.jl")
include("standard_form.jl")

export LinearModel, QuadraticModel, LPData, QPData
export BatchQuadraticModel, ObjRHSBatchQuadraticModel, UniformBatchQuadraticModel, batch_model
export BatchSparseOperator, batch_spmv!, batch_spmv_subset!
export batch_mapreduce!, batch_maximum!, batch_minimum!, batch_sum!
export gather_columns!, gather_entries!
export obj_subset!, grad_subset!, cons_subset!, jac_coord_subset!, hess_coord_subset!
export standard_form, update_standard_form!
export recover_primal!, recover_primal, recover_variable_multipliers!
export StandardFormWorkspace

end # module
