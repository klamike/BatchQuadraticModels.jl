mutable struct SparseOperator{T, M <: AbstractMatrix{T}, M2 <: AbstractMatrix{T}} <: AbstractSparseOperator{T}
  m::Int
  n::Int
  A::M
  mat::M2
  transa::Char
end

Base.eltype(A::SparseOperator{T}) where {T} = T
Base.size(A::SparseOperator) = (A.m, A.n)
SparseArrays.nnz(A::SparseOperator) = nnz(A.A)

operator_sparse_matrix(A::SparseMatrixCOO) = A
operator_sparse_matrix(A::SparseMatrixCSC) = A
operator_sparse_matrix(A::SparseOperator) = A.A

_csc_matrix(A::SparseMatrixCSC) = A
_csc_matrix(A::SparseMatrixCOO) = sparse(A.rows, A.cols, A.vals, size(A)...)

sparse_operator(A::AbstractSparseOperator; transa::Char = 'N', symmetric::Bool = false) = A

function sparse_operator(A::SparseMatrixCSC{T}; transa::Char = 'N', symmetric::Bool = false) where {T}
  @assert transa == 'N' "Only non-transposed sparse operators are supported on CPU"
  mat = symmetric ? Symmetric(A, :L) : A
  return SparseOperator{T, typeof(A), typeof(mat)}(size(A, 1), size(A, 2), A, mat, transa)
end

function sparse_operator(A::SparseMatrixCOO{T}; transa::Char = 'N', symmetric::Bool = false) where {T}
  @assert transa == 'N' "Only non-transposed sparse operators are supported on CPU"
  A_csc = _csc_matrix(A)
  mat = symmetric ? Symmetric(A_csc, :L) : A_csc
  return SparseOperator{T, typeof(A), typeof(mat)}(size(A, 1), size(A, 2), A, mat, transa)
end

function Adapt.adapt_structure(to, A::SparseOperator{T}) where {T}
  A_adapted = Adapt.adapt(to, A.A)
  mat_adapted = Adapt.adapt(to, A.mat)
  return SparseOperator{T, typeof(A_adapted), typeof(mat_adapted)}(A.m, A.n, A_adapted, mat_adapted, A.transa)
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, A::SparseOperator{T}, X::AbstractMatrix{T}) where {T}
  mul!(Y, A.mat, X)
  return Y
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::SparseOperator{T}, x::AbstractVector{T}) where {T}
  mul!(y, A.mat, x)
  return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, A::SparseOperator{T}, X::AbstractMatrix{T}, α::Number, β::Number) where {T}
  mul!(Y, A.mat, X, α, β)
  return Y
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::SparseOperator{T}, x::AbstractVector{T}, α::Number, β::Number) where {T}
  mul!(y, A.mat, x, α, β)
  return y
end
