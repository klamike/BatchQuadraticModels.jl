function BatchQuadraticModels._copy_sparse_values!(A::Union{CUSPARSE.CuSparseMatrixCSR, CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCOO}, vals::CuVector)
  @assert length(vals) == nnz(A)
  copyto!(vals, A.nzVal)
  return vals
end

BatchQuadraticModels._copy_sparse_structure!(A::CuSparseOperator, rows::CuVector{<:Integer}, cols::CuVector{<:Integer}) =
  BatchQuadraticModels._copy_sparse_structure!(operator_sparse_matrix(A), rows, cols)

BatchQuadraticModels._copy_sparse_values!(A::CuSparseOperator, vals::CuVector) = BatchQuadraticModels._copy_sparse_values!(operator_sparse_matrix(A), vals)

function _mul_jt!(jtv, A::CuSparseOperator{T}, v) where {T}
  mul!(jtv, transpose(A), v)
  return jtv
end

function Adapt.adapt_structure(
  to,
  data::LPData{T, VT, SparseMatrixCSC{T, Int}},
) where {T, VT}
  A_gpu = sparse_operator(CUSPARSE.CuSparseMatrixCSR(data.A); symmetric = false)
  return LPData(
    A_gpu,
    Adapt.adapt(to, data.c);
    lcon = Adapt.adapt(to, data.lcon),
    ucon = Adapt.adapt(to, data.ucon),
    lvar = Adapt.adapt(to, data.lvar),
    uvar = Adapt.adapt(to, data.uvar),
    c0 = data.c0[],
  )
end

function Adapt.adapt_structure(
  to,
  data::QPData{T, VT, SparseMatrixCSC{T, Int}, SparseMatrixCSC{T, Int}},
) where {T, VT}
  Q_gpu = sparse_operator(CUSPARSE.CuSparseMatrixCSR(data.Q); symmetric = true)
  A_gpu = sparse_operator(CUSPARSE.CuSparseMatrixCSR(data.A); symmetric = false)
  return QPData(
    A_gpu,
    Adapt.adapt(to, data.c),
    Q_gpu;
    lcon = Adapt.adapt(to, data.lcon),
    ucon = Adapt.adapt(to, data.ucon),
    lvar = Adapt.adapt(to, data.lvar),
    uvar = Adapt.adapt(to, data.uvar),
    c0 = data.c0[],
    _v = Adapt.adapt(to, data._v),
  )
end

function Adapt.adapt_structure(
  to,
  lp::LinearModel{T, VT, SparseMatrixCSC{T, Int}},
) where {T, VT}
  data_gpu = Adapt.adapt(to, lp.data)
  return LinearModel(
    data_gpu;
    x0 = Adapt.adapt(to, lp.meta.x0),
    y0 = Adapt.adapt(to, lp.meta.y0),
    minimize = lp.meta.minimize,
    name = lp.meta.name,
  )
end

function Adapt.adapt_structure(
  to,
  qp::QuadraticModel{T, VT, SparseMatrixCSC{T, Int}, SparseMatrixCSC{T, Int}},
) where {T, VT}
  data_gpu = Adapt.adapt(to, qp.data)
  return QuadraticModel(
    data_gpu;
    x0 = Adapt.adapt(to, qp.meta.x0),
    y0 = Adapt.adapt(to, qp.meta.y0),
    minimize = qp.meta.minimize,
    name = qp.meta.name,
  )
end
