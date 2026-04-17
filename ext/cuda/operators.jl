mutable struct CuSparseOperator{T, S, O} <: AbstractSparseOperator{T}
  m::Int
  n::Int
  source::S
  op::O
  descA::CUSPARSE.CuSparseMatrixDescriptor
  buffer_N::Union{Nothing, CuVector{UInt8}}
  buffer_T::Union{Nothing, CuVector{UInt8}}
  spmm_buffer_N::Union{Nothing, CuVector{UInt8}}
  spmm_buffer_T::Union{Nothing, CuVector{UInt8}}
end

Base.eltype(A::CuSparseOperator{T}) where {T} = T
Base.size(A::CuSparseOperator) = (A.m, A.n)
SparseArrays.nnz(A::CuSparseOperator) = nnz(A.source)
operator_sparse_matrix(A::CuSparseOperator) = A.source

function _coo_to_cu_csr(A::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
  return CUSPARSE.CuSparseMatrixCSR(sparse(A.rows, A.cols, A.vals, size(A)...))
end

_to_cu_csr(A::SparseMatrixCOO) = _coo_to_cu_csr(A)
_to_cu_csr(A::SparseMatrixCSC) = CUSPARSE.CuSparseMatrixCSR(A)

function _expand_symmetric_matrix(H::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
  rows, cols, vals = H.rows, H.cols, H.vals
  offdiag = findall(i -> rows[i] != cols[i], eachindex(rows))
  return SparseMatrixCOO(size(H, 1), size(H, 2), vcat(rows, cols[offdiag]), vcat(cols, rows[offdiag]), vcat(vals, vals[offdiag]))
end

_expand_symmetric_matrix(H::SparseMatrixCSC) = sparse(Symmetric(H, :L))

_mode_field(transa::Char) = transa == 'N' ? :buffer_N : transa == 'T' ? :buffer_T : throw(ArgumentError("invalid sparse operator mode $transa"))
_spmm_mode_field(transa::Char) = transa == 'N' ? :spmm_buffer_N : transa == 'T' ? :spmm_buffer_T : throw(ArgumentError("invalid sparse operator mode $transa"))

function _normalize_modes(modes)
  normalized = Char[]
  for mode in modes
    mode′ = Char(mode)
    mode′ in ('N', 'T') || throw(ArgumentError("invalid sparse operator mode $mode′"))
    mode′ in normalized || push!(normalized, mode′)
  end
  return normalized
end

_normalize_modes(::Nothing) = Char[]

function _spmv_buffer(descA, ::Type{T}, m::Int, n::Int, transa::Char) where {T <: BlasFloat}
  xdim, ydim = transa == 'N' ? (n, m) : (m, n)
  alpha = Ref{T}(one(T))
  beta = Ref{T}(zero(T))
  descX = CUSPARSE.CuDenseVectorDescriptor(T, xdim)
  descY = CUSPARSE.CuDenseVectorDescriptor(T, ydim)
  algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
  buffer_size = Ref{Csize_t}()
  CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer_size)
  buffer = CuVector{UInt8}(undef, buffer_size[])
  if CUSPARSE.version() ≥ v"12.3"
    CUSPARSE.cusparseSpMV_preprocess(CUSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer)
  end
  return buffer
end

function _spmm_buffer(descA, ::Type{T}, m::Int, n::Int, transa::Char, spmm_ncols::Int) where {T <: BlasFloat}
  spmm_ncols > 0 || return CuVector{UInt8}(undef, 0)
  xrows, yrows = transa == 'N' ? (n, m) : (m, n)
  alpha = Ref{T}(one(T))
  beta = Ref{T}(zero(T))
  descB = CUSPARSE.CuDenseMatrixDescriptor(T, xrows, spmm_ncols)
  descC = CUSPARSE.CuDenseMatrixDescriptor(T, yrows, spmm_ncols)
  spmm_buf_size = Ref{Csize_t}()
  spmm_algo = CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT
  CUSPARSE.cusparseSpMM_bufferSize(CUSPARSE.handle(), transa, 'N', alpha, descA, descB, beta, descC, T, spmm_algo, spmm_buf_size)
  buffer = CuVector{UInt8}(undef, spmm_buf_size[])
  if CUSPARSE.version() ≥ v"12.3"
    CUSPARSE.cusparseSpMM_preprocess(CUSPARSE.handle(), transa, 'N', alpha, descA, descB, beta, descC, T, spmm_algo, buffer)
  end
  return buffer
end

function _spmv_buffer!(op::CuSparseOperator{T}, transa::Char) where {T <: BlasFloat}
  field = _mode_field(transa)
  buffer = getfield(op, field)
  if isnothing(buffer)
    buffer = _spmv_buffer(op.descA, T, op.m, op.n, transa)
    setfield!(op, field, buffer)
  end
  return buffer
end

function _spmm_buffer(op::CuSparseOperator, transa::Char)
  buffer = getfield(op, _spmm_mode_field(transa))
  isnothing(buffer) && throw(ArgumentError("SpMM buffer for mode $transa was not preallocated; pass `premake_spmm` when constructing the operator."))
  return buffer
end

function _mul!(y::CuVector{T}, A::CuSparseOperator{T}, x::CuVector{T}, transa::Char, α::T, β::T) where {T <: BlasFloat}
  expected_y, expected_x = transa == 'N' ? (A.m, A.n) : (A.n, A.m)
  length(y) == expected_y || throw(DimensionMismatch("length(y) != $expected_y"))
  length(x) == expected_x || throw(DimensionMismatch("length(x) != $expected_x"))
  alpha = Ref{T}(α)
  beta = Ref{T}(β)
  descY = CUSPARSE.CuDenseVectorDescriptor(y)
  descX = CUSPARSE.CuDenseVectorDescriptor(x)
  algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
  CUSPARSE.cusparseSpMV(CUSPARSE.handle(), transa, alpha, A.descA, descX, beta, descY, T, algo, _spmv_buffer!(A, transa))
  return y
end

function _mul!(Y::CuMatrix{T}, A::CuSparseOperator{T}, X::CuMatrix{T}, transa::Char, α::T, β::T) where {T <: BlasFloat}
  expected_y, expected_x = transa == 'N' ? (A.m, A.n) : (A.n, A.m)
  size(Y, 1) == expected_y || throw(DimensionMismatch("size(Y,1) != $expected_y"))
  size(X, 1) == expected_x || throw(DimensionMismatch("size(X,1) != $expected_x"))
  alpha = Ref{T}(α)
  beta = Ref{T}(β)
  descX = CUSPARSE.CuDenseMatrixDescriptor(X)
  descY = CUSPARSE.CuDenseMatrixDescriptor(Y)
  CUSPARSE.cusparseSpMM(CUSPARSE.handle(), transa, 'N', alpha, A.descA, descX, beta, descY, T, CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT, _spmm_buffer(A, transa))
  return Y
end

function _cu_sparse_operator(source, op; spmm_ncols::Int = 0, premake_spmv = ('N',), premake_spmm = ('N',))
  T = eltype(op)
  m, n = size(op)
  descA = CUSPARSE.CuSparseMatrixDescriptor(op, 'O')
  spmv_modes = _normalize_modes(premake_spmv)
  spmm_modes = spmm_ncols > 0 ? _normalize_modes(premake_spmm) : Char[]
  buffer_N = 'N' in spmv_modes ? _spmv_buffer(descA, T, m, n, 'N') : nothing
  buffer_T = 'T' in spmv_modes ? _spmv_buffer(descA, T, m, n, 'T') : nothing
  spmm_buffer_N = spmm_ncols > 0 && 'N' in spmm_modes ? _spmm_buffer(descA, T, m, n, 'N', spmm_ncols) : nothing
  spmm_buffer_T = spmm_ncols > 0 && 'T' in spmm_modes ? _spmm_buffer(descA, T, m, n, 'T', spmm_ncols) : nothing
  return CuSparseOperator{T, typeof(source), typeof(op)}(m, n, source, op, descA, buffer_N, buffer_T, spmm_buffer_N, spmm_buffer_T)
end

const _CuSparseMatrix{T} = Union{CuSparseMatrixCSR{T}, CuSparseMatrixCSC{T}, CuSparseMatrixCOO{T}}

function sparse_operator(
  A::_CuSparseMatrix{T};
  transa::Char = 'N',
  symmetric::Bool = false,
  spmm_ncols::Int = 0,
  premake_spmv = ('N',),
  premake_spmm = ('N',),
) where {T <: BlasFloat}
  op = symmetric && nnz(A) > 0 ? tril(A, -1) + A' : A
  sparse = _cu_sparse_operator(A, op; spmm_ncols = spmm_ncols, premake_spmv = premake_spmv, premake_spmm = spmm_ncols > 0 ? (premake_spmm..., transa) : premake_spmm)
  return transa == 'N' ? sparse : transpose(sparse)
end

const _CuArrT{T} = Union{CuMatrix{T}, CuVector{T}}

LinearAlgebra.mul!(y::_CuArrT{T}, A::CuSparseOperator{T}, x::_CuArrT{T}, α::Number, β::Number) where {T <: BlasFloat} =
  _mul!(y, A, x, 'N', T(α), T(β))
LinearAlgebra.mul!(y::_CuArrT{T}, A::CuSparseOperator{T}, x::_CuArrT{T}) where {T <: BlasFloat} =
  _mul!(y, A, x, 'N', one(T), zero(T))
LinearAlgebra.mul!(y::_CuArrT{T}, At::Transpose{T, <:CuSparseOperator{T}}, x::_CuArrT{T}, α::Number, β::Number) where {T <: BlasFloat} =
  _mul!(y, parent(At), x, 'T', T(α), T(β))
LinearAlgebra.mul!(y::_CuArrT{T}, At::Transpose{T, <:CuSparseOperator{T}}, x::_CuArrT{T}) where {T <: BlasFloat} =
  _mul!(y, parent(At), x, 'T', one(T), zero(T))
