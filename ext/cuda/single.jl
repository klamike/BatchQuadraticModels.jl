@kernel function _fill_sparse_structure!(rows, cols, Ap, Aj)
  i = @index(Global, Linear)
  for c in Ap[i]:Ap[i + 1] - 1
    rows[c] = i
    cols[c] = Aj[c]
  end
end

function _copy_sparse_structure!(A::CUSPARSE.CuSparseMatrixCSR, rows::CuVector, cols::CuVector)
  @assert length(cols) == length(rows)
  length(cols) > 0 && _fill_sparse_structure!(CUDABackend())(rows, cols, A.rowPtr, A.colVal; ndrange = size(A, 1))
  return rows, cols
end

function _copy_sparse_structure!(A::CUSPARSE.CuSparseMatrixCOO, rows::CuVector, cols::CuVector)
  @assert length(cols) == length(rows) == nnz(A)
  copyto!(rows, A.rowInd)
  copyto!(cols, A.colInd)
  return rows, cols
end

function _copy_sparse_values!(A::Union{CUSPARSE.CuSparseMatrixCSR, CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCOO}, vals::CuVector)
  @assert length(vals) == nnz(A)
  copyto!(vals, A.nzVal)
  return vals
end

_copy_sparse_structure!(A::CuSparseOperator, rows::CuVector, cols::CuVector) =
  _copy_sparse_structure!(operator_sparse_matrix(A), rows, cols)

_copy_sparse_values!(A::CuSparseOperator, vals::CuVector) =
  _copy_sparse_values!(operator_sparse_matrix(A), vals)

function _mul_jt!(jtv, A::CuSparseOperator{T}, v) where {T}
  mul!(jtv, transpose(A), v)
  return jtv
end

# LP/QP data and models always wrap A/Q in `SparseOperator` (see `LPData`/`QPData`
# ctors). Adapt the wrapped source CSC to GPU, rebuild via `sparse_operator`.

_adapt_data_op(to, op::SparseOperator; symmetric::Bool) =
  sparse_operator(CUSPARSE.CuSparseMatrixCSR(operator_sparse_matrix(op)); symmetric)

function Adapt.adapt_structure(to, data::LPData{T, VT, <:SparseOperator}) where {T, VT}
  return LPData(
    _adapt_data_op(to, data.A; symmetric = false),
    Adapt.adapt(to, data.c);
    lcon = Adapt.adapt(to, data.lcon), ucon = Adapt.adapt(to, data.ucon),
    lvar = Adapt.adapt(to, data.lvar), uvar = Adapt.adapt(to, data.uvar),
    c0 = data.c0[],
  )
end

function Adapt.adapt_structure(to, data::QPData{T, VT, <:SparseOperator, <:SparseOperator}) where {T, VT}
  return QPData(
    _adapt_data_op(to, data.A; symmetric = false),
    Adapt.adapt(to, data.c),
    _adapt_data_op(to, data.Q; symmetric = true);
    lcon = Adapt.adapt(to, data.lcon), ucon = Adapt.adapt(to, data.ucon),
    lvar = Adapt.adapt(to, data.lvar), uvar = Adapt.adapt(to, data.uvar),
    c0 = data.c0[], _v = Adapt.adapt(to, data._v),
  )
end

# Splat into a model ctor: adapts x0/y0 to `to`; minimize/name are bits.
_adapt_meta_kwargs(to, meta) = (x0 = Adapt.adapt(to, meta.x0), y0 = Adapt.adapt(to, meta.y0),
                                minimize = meta.minimize, name = meta.name)

Adapt.adapt_structure(to, lp::LinearModel{T, VT, <:SparseOperator}) where {T, VT} =
  LinearModel(Adapt.adapt(to, lp.data); _adapt_meta_kwargs(to, lp.meta)...)
Adapt.adapt_structure(to, qp::QuadraticModel{T, VT, <:SparseOperator, <:SparseOperator}) where {T, VT} =
  QuadraticModel(Adapt.adapt(to, qp.data); _adapt_meta_kwargs(to, qp.meta)...)
