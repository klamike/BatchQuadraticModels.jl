using Test
using LinearAlgebra
using SparseArrays

using NLPModels
using QuadraticModels
using BatchQuadraticModels

const SparseMatrixCOO = QuadraticModels.SparseMatrixCOO

function ineqconqp_QP()
  c = -ones(2)
  Hrows = [1, 2]
  Hcols = [1, 2]
  Hvals = ones(2)
  Arows = [1, 1, 2, 2, 3, 3]
  Acols = [1, 2, 1, 2, 1, 2]
  Avals = [1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
  c0 = 1.0
  lcon = [0.0, -Inf, -1.0]
  ucon = [Inf, 0.0, 1.0]
  x0 = ones(2)

  return QuadraticModel(
    c,
    Hrows,
    Hcols,
    Hvals,
    Arows = Arows,
    Acols = Acols,
    Avals = Avals,
    lcon = lcon,
    ucon = ucon,
    c0 = c0,
    x0 = x0,
    name = "ineqconqp_QP",
  )
end

include("objrhsbatchqp.jl")
include("batchqp.jl")
include("cuda.jl")
