"""
    SparseOperator{T, S, O} <: AbstractSparseOperator{T}

Scalar sparse-matrix wrapper. `source::S` is the user-supplied sparse matrix
(`SparseMatrixCSC` or `SparseMatrixCOO`); `op::O` is the backend-ready form
(`SparseMatrixCSC` for non-symmetric A, `Symmetric{SparseMatrixCSC}` for
Hessian Q). `mul!` forwards to `op`; structural helpers (`nnz`, `_sparse_*`)
forward to `source`.

The GPU extension adds a mutable `CuSparseOperator` with the same shape plus
CUSPARSE buffer caches for SpMV/SpMM.
"""
struct SparseOperator{T, S, O} <: AbstractSparseOperator{T}
  source::S
  op::O
end

Base.size(A::SparseOperator) = size(A.source)
SparseArrays.nnz(A::SparseOperator) = nnz(A.source)

# Unwrap to the underlying sparse matrix. Direct `SparseMatrixCSC`/`COO` pass
# through so the same callers work with or without a `SparseOperator` wrap.
operator_sparse_matrix(A::SparseMatrixCOO) = A
operator_sparse_matrix(A::SparseMatrixCSC) = A
operator_sparse_matrix(A::SparseOperator)  = A.source

# COO → CSC conversion; `mul!(transpose(_), v)` lacks a COO method, and
# `sparse_operator` builds its `op` field from CSC either way.
_csc_matrix(A::SparseMatrixCSC) = A
_csc_matrix(A::SparseMatrixCOO) = sparse(A.rows, A.cols, A.vals, size(A)...)

"""
    sparse_operator(A; transa='N', symmetric=false)

Wrap `A` as a [`SparseOperator`](@ref). `symmetric=true` uses the
lower-triangular `Symmetric` view for Hessian operands. `transa='T'` returns
a lazy `transpose(SparseOperator)` — `mul!` then dispatches through
`Base`'s transpose-wrapper handling.
"""
sparse_operator(A::AbstractSparseOperator; transa::Char = 'N', symmetric::Bool = false) =
  transa == 'N' ? A : transpose(A)

function sparse_operator(A::Union{SparseMatrixCSC{T}, SparseMatrixCOO{T}};
                         transa::Char = 'N', symmetric::Bool = false) where {T}
  A_csc = _csc_matrix(A)
  op = symmetric ? Symmetric(A_csc, :L) : A_csc
  wrapped = SparseOperator{T, typeof(A), typeof(op)}(A, op)
  return transa == 'N' ? wrapped : transpose(wrapped)
end

function Adapt.adapt_structure(to, A::SparseOperator{T}) where {T}
  source = Adapt.adapt(to, A.source)
  op = Adapt.adapt(to, A.op)
  return SparseOperator{T, typeof(source), typeof(op)}(source, op)
end

# `mul!` forwards to the backing `op`. Only the 5-arg form is defined; Base
# routes the 3-arg `mul!(y, A, x)` through `mul!(y, A, x, true, false)`.
LinearAlgebra.mul!(Y::AbstractMatrix{T}, A::SparseOperator{T}, X::AbstractMatrix{T}, α::Number, β::Number) where {T} = mul!(Y, A.op, X, α, β)
LinearAlgebra.mul!(y::AbstractVector{T}, A::SparseOperator{T}, x::AbstractVector{T}, α::Number, β::Number) where {T} = mul!(y, A.op, x, α, β)
