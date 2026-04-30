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
const BoundSet = Union{
  MOI.LessThan{Float64},
  MOI.GreaterThan{Float64},
  MOI.Interval{Float64},
  MOI.EqualTo{Float64},
}

include("moi/qp_model.jl")

end # module BatchQuadraticModelsMathOptInterfaceExt
