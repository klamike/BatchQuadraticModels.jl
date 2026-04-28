# Each problem datum A / Q on a `BatchQuadraticModel` is stored as either a
# scalar `SparseOperator` (one copy shared across the batch — ObjRHS shape) or
# a `BatchSparseOperator` (per-instance nzvals stacked with a batched operator
# — Uniform shape). Both share `mul!`, `_sparse_structure`, `_sparse_values`,
# `_copy_sparse_*`, and `_weighted_sparse_values!` via existing dispatches —
# callers don't branch on kind.

# ---------- entry builders ----------

# `Varying` entry: stack per-instance nzvals and build a batched operator.
# Hessian uses symmetric expansion so SpMV against the symmetric-layout
# operator yields the full `Q x`.
function _varying_op(nzvals, rows, cols, scatter_rows, scatter_nz, scatter_cols, nrow::Int)
  rowptr, colidx = _coo_to_csr(scatter_rows, nrow)
  return _build_op(nzvals, rows, cols, rowptr, scatter_nz, scatter_cols, colidx)
end

# A (no symmetry, scatter is identity) and Q (symmetric-expanded) builders.
# `nzvals` is the caller-supplied per-instance value matrix (stacked or replicated).
function _jacobian_op(qp_ref, nzvals)
  rows, cols = _sparse_structure(qp_ref.data.A)
  id = _indices_like(rows, qp_ref.meta.nnzj)
  return _varying_op(nzvals, rows, cols, rows, id, cols, qp_ref.meta.ncon)
end

function _hessian_op(qp_ref, nzvals)
  rows, cols = _sparse_structure(qp_ref.data.Q)
  sym_rows, sym_nz, sym_cols = _symmetric_scatter_ops(rows, cols, qp_ref.meta.nnzh)
  return _varying_op(nzvals, rows, cols, sym_rows, sym_nz, sym_cols, qp_ref.meta.nvar)
end

# ---------- structural comparisons (for auto-selecting Shared vs Varying) ----
# Per-type tuple/seq accessors; cross-type pairs short-circuit via typeof check.

_struct_tuple(A::SparseMatrixCOO) = (size(A), A.rows, A.cols)
_struct_tuple(A::SparseMatrixCSC) = (size(A), A.colptr, rowvals(A))
_value_seq(A::SparseMatrixCOO)    = A.vals
_value_seq(A::SparseMatrixCSC)    = nonzeros(A)

@inline _unwrap(A::AbstractSparseOperator) = operator_sparse_matrix(A)
@inline _unwrap(A) = A

function _same_matrix_structure(A, B)
  A === B && return true
  a, b = _unwrap(A), _unwrap(B)
  typeof(a) === typeof(b) && _struct_tuple(a) == _struct_tuple(b)
end
function _same_matrix_values(A, B)
  A === B && return true
  _same_matrix_structure(A, B) && _value_seq(_unwrap(A)) == _value_seq(_unwrap(B))
end

function _batch_traits(qps::Vector{<:QuadraticModel})
  @assert !isempty(qps) "Need at least one model"
  qp1     = first(qps)
  same_A_values = true
  same_Q_values = true
  same_A_structure = true
  same_Q_structure = true
  for qp in @view qps[2:end]
    @assert qp.meta.nvar == qp1.meta.nvar         "All models must have same nvar"
    @assert qp.meta.ncon == qp1.meta.ncon         "All models must have same ncon"
    @assert qp.meta.nnzj == qp1.meta.nnzj         "All models must have same nnzj"
    @assert qp.meta.nnzh == qp1.meta.nnzh         "All models must have same nnzh"
    @assert qp.meta.minimize == qp1.meta.minimize "All models must have the same objective sense"
    same_A_values    &= _same_matrix_values(qp.data.A, qp1.data.A)
    same_Q_values    &= _same_matrix_values(qp.data.Q, qp1.data.Q)
    same_A_structure &= _same_matrix_structure(qp.data.A, qp1.data.A)
    same_Q_structure &= _same_matrix_structure(qp.data.Q, qp1.data.Q)
  end
  objrhs = same_A_values && same_Q_values
  uniform = same_A_structure && same_Q_structure
  return (; same_A_values, same_Q_values, same_A_structure, same_Q_structure, objrhs, uniform)
end

"""
    batch_model(qps; validate = false, MT = nothing, name)

Build a `BatchQuadraticModel` from `qps`, auto-selecting `Shared`/`Varying`
storage for A and Q based on whether nonzero values match across the batch.
"""
function batch_model(qps::Vector{<:QuadraticModel};
                     name::String = "batch_model",
                     validate::Bool = false, MT = nothing)
  traits = _batch_traits(qps)
  traits.uniform || error("Unable to select a batch model: the batch does not share a common sparsity structure")
  return BatchQuadraticModel(qps; shared_A = traits.objrhs, shared_Q = traits.objrhs,
                              name, validate, MT)
end
