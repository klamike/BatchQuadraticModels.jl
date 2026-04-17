const BatchLinearModel = BatchQuadraticModel
const ObjRHSBatchLinearModel = ObjRHSBatchQuadraticModel

function _same_matrix_structure(A::SparseMatrixCOO, B::SparseMatrixCOO)
  return size(A) == size(B) && A.rows == B.rows && A.cols == B.cols
end

function _same_matrix_values(A::SparseMatrixCOO, B::SparseMatrixCOO)
  return _same_matrix_structure(A, B) && A.vals == B.vals
end

function _same_matrix_structure(A::SparseMatrixCSC, B::SparseMatrixCSC)
  return size(A) == size(B) && A.colptr == B.colptr && rowvals(A) == rowvals(B)
end

function _same_matrix_values(A::SparseMatrixCSC, B::SparseMatrixCSC)
  return _same_matrix_structure(A, B) && nonzeros(A) == nonzeros(B)
end

_same_matrix_structure(A, B) = false
_same_matrix_values(A, B) = false

function _check_batch_compatibility(qps::Vector{QP}) where {QP <: QuadraticModel}
  nbatch = length(qps)
  @assert nbatch > 0 "Need at least one model"
  qp1 = first(qps)
  for qp in qps[2:end]
    @assert qp.meta.nvar == qp1.meta.nvar "All models must have same nvar"
    @assert qp.meta.ncon == qp1.meta.ncon "All models must have same ncon"
    @assert qp.meta.nnzj == qp1.meta.nnzj "All models must have same nnzj"
    @assert qp.meta.nnzh == qp1.meta.nnzh "All models must have same nnzh"
    @assert qp.meta.minimize == qp1.meta.minimize "All models must have the same objective sense"
  end
  return qp1
end

function _batch_traits(qps::Vector{QP}) where {QP <: QuadraticModel}
  qp1 = _check_batch_compatibility(qps)
  objrhs = _objrhs_matrix_supported(qp1.data.Q) && _objrhs_matrix_supported(qp1.data.A)
  uniform = true
  islp = qp1.meta.nnzh == 0
  for qp in qps[2:end]
    objrhs &= qp.data.c0 == qp1.data.c0 &&
              _same_matrix_values(qp.data.Q, qp1.data.Q) &&
              _same_matrix_values(qp.data.A, qp1.data.A)
    uniform &= _same_matrix_structure(qp.data.Q, qp1.data.Q) &&
               _same_matrix_structure(qp.data.A, qp1.data.A)
    islp &= qp.meta.nnzh == 0
  end
  return (; objrhs, uniform, islp)
end

function _validate_uniform_batch(qps::Vector{QP}, model_name::AbstractString) where {QP <: QuadraticModel}
  @assert _batch_traits(qps).uniform "$model_name requires identical sparse structure across the batch; pass validate=false to skip this check"
  return qps
end

function _validate_objrhs_batch(qps::Vector{QP}, model_name::AbstractString) where {QP <: QuadraticModel}
  @assert _batch_traits(qps).objrhs "$model_name requires shared static data across the batch; pass validate=false to skip this check"
  return qps
end

function batch_model(
  qps::Vector{QP};
  name::String = begin
    traits = _batch_traits(qps)
    traits.objrhs ? (traits.islp ? "ObjRHSBatchLP" : "ObjRHSBatchQP") : (traits.islp ? "SameStructBatchLP" : "SameStructBatchQP")
  end,
  validate::Bool = false,
  MT = nothing,
) where {QP <: QuadraticModel}
  traits = _batch_traits(qps)
  if traits.objrhs
    return ObjRHSBatchQuadraticModel(qps; name = name, validate = validate, MT = MT)
  end
  if traits.uniform
    return BatchQuadraticModel(qps; name = name, validate = validate, MT = MT)
  end
  error("Unable to select a batch model: the batch does not share common static data or a common sparsity structure")
end
