struct SparseOperator{T, S, O} <: AbstractSparseOperator{T}
  m::Int
  n::Int
  source::S
  op::O
end

Base.eltype(A::SparseOperator{T}) where {T} = T
Base.size(A::SparseOperator) = (A.m, A.n)
SparseArrays.nnz(A::SparseOperator) = nnz(A.source)

operator_sparse_matrix(A::SparseMatrixCOO) = A
operator_sparse_matrix(A::SparseMatrixCSC) = A
operator_sparse_matrix(A::SparseOperator) = A.source

_csc_matrix(A::SparseMatrixCSC) = A
_csc_matrix(A::SparseMatrixCOO) = sparse(A.rows, A.cols, A.vals, size(A)...)

sparse_operator(A::AbstractSparseOperator; transa::Char = 'N', symmetric::Bool = false) =
  transa == 'N' ? A : transpose(A)

function _cpu_sparse_operator(A, op)
  T = eltype(op)
  return SparseOperator{T, typeof(A), typeof(op)}(size(A, 1), size(A, 2), A, op)
end

function sparse_operator(A::SparseMatrixCSC{T}; transa::Char = 'N', symmetric::Bool = false) where {T}
  op = symmetric ? Symmetric(A, :L) : A
  return transa == 'N' ? _cpu_sparse_operator(A, op) : transpose(_cpu_sparse_operator(A, op))
end

function sparse_operator(A::SparseMatrixCOO{T}; transa::Char = 'N', symmetric::Bool = false) where {T}
  A_csc = _csc_matrix(A)
  op = symmetric ? Symmetric(A_csc, :L) : A_csc
  return transa == 'N' ? _cpu_sparse_operator(A, op) : transpose(_cpu_sparse_operator(A, op))
end

function Adapt.adapt_structure(to, A::SparseOperator{T}) where {T}
  source = Adapt.adapt(to, A.source)
  op = Adapt.adapt(to, A.op)
  return SparseOperator{T, typeof(source), typeof(op)}(A.m, A.n, source, op)
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, A::SparseOperator{T}, X::AbstractMatrix{T}) where {T}
  mul!(Y, A.op, X)
  return Y
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::SparseOperator{T}, x::AbstractVector{T}) where {T}
  mul!(y, A.op, x)
  return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, A::SparseOperator{T}, X::AbstractMatrix{T}, α::Number, β::Number) where {T}
  mul!(Y, A.op, X, α, β)
  return Y
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::SparseOperator{T}, x::AbstractVector{T}, α::Number, β::Number) where {T}
  mul!(y, A.op, x, α, β)
  return y
end
