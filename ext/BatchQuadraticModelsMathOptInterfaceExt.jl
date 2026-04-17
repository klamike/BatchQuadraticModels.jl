module BatchQuadraticModelsMathOptInterfaceExt

using MathOptInterface
using SparseArrays
import BatchQuadraticModels
import BatchQuadraticModels: QPData, QuadraticModel, qp_model

const MOI = MathOptInterface
const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction{Float64}
const VAF = MOI.VectorAffineFunction{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}
const AF = Union{SAF, VAF}
const LinQuad = Union{VI, SAF, SQF}

include("moi/qp_model.jl")

end # module BatchQuadraticModelsMathOptInterfaceExt
