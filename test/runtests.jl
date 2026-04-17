using Test
using Adapt
using LinearAlgebra
using SparseArrays

using NLPModels
using SparseMatricesCOO
using BatchQuadraticModels
import BatchQuadraticModels: QuadraticModel

const SparseMatrixCOO = SparseMatricesCOO.SparseMatrixCOO

@test !Base.ismutabletype(LPData)
@test !Base.ismutabletype(QPData)

function QuadraticModel(
  c::AbstractVector{T},
  Q::AbstractMatrix{T};
  A = spzeros(T, 0, length(c)),
  lcon = fill(T(-Inf), size(A, 1)),
  ucon = fill(T(Inf), size(A, 1)),
  lvar = fill(T(-Inf), length(c)),
  uvar = fill(T(Inf), length(c)),
  c0 = zero(T),
  x0 = zeros(T, length(c)),
  y0 = zeros(T, size(A, 1)),
  minimize::Bool = true,
  kwargs...,
) where {T}
  data = QPData(A, collect(c), Q; lcon = collect(lcon), ucon = collect(ucon), lvar = collect(lvar), uvar = collect(uvar), c0 = c0)
  return BatchQuadraticModels.QuadraticModel(data; x0 = collect(x0), y0 = collect(y0), minimize = minimize, kwargs...)
end

function QuadraticModel(
  c::AbstractVector{T},
  Hrows::AbstractVector{<:Integer},
  Hcols::AbstractVector{<:Integer},
  Hvals::AbstractVector{T};
  Arows::AbstractVector{<:Integer} = Int[],
  Acols::AbstractVector{<:Integer} = Int[],
  Avals::AbstractVector{T} = T[],
  lcon::AbstractVector{T} = T[],
  ucon::AbstractVector{T} = T[],
  lvar::AbstractVector{T} = fill(T(-Inf), length(c)),
  uvar::AbstractVector{T} = fill(T(Inf), length(c)),
  c0 = zero(T),
  x0 = zeros(T, length(c)),
  y0 = T[],
  minimize::Bool = true,
  kwargs...,
) where {T}
  nvar = length(c)
  ncon = max(length(lcon), length(ucon), isempty(Arows) ? 0 : maximum(Arows))
  H = SparseMatrixCOO(nvar, nvar, Vector{Int}(Hrows), Vector{Int}(Hcols), collect(Hvals))
  A = SparseMatrixCOO(ncon, nvar, Vector{Int}(Arows), Vector{Int}(Acols), collect(Avals))
  y0v = length(y0) == ncon ? collect(y0) : zeros(T, ncon)
  return QuadraticModel(
    collect(c),
    H;
    A = A,
    lcon = isempty(lcon) ? fill(T(-Inf), ncon) : collect(lcon),
    ucon = isempty(ucon) ? fill(T(Inf), ncon) : collect(ucon),
    lvar = collect(lvar),
    uvar = collect(uvar),
    c0 = c0,
    x0 = collect(x0),
    y0 = y0v,
    minimize = minimize,
    kwargs...,
  )
end

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
include("batch_model.jl")
include("cuda.jl")
