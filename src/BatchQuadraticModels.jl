module BatchQuadraticModels

using Adapt
using LinearAlgebra, SparseArrays
using NLPModels
using SparseMatricesCOO

abstract type AbstractSparseOperator{T} <: AbstractMatrix{T} end

function sparse_operator end
function operator_sparse_matrix end
function qp_model end

_resolve_batch_matrix_type(qp, ::Type{T}, MT) where {T} = MT === nothing ? typeof(similar(qp.data.c, T, 0, 0)) : MT

function _structure_arrays(A::SparseMatrixCOO)
  return Vector{Int}(A.rows), Vector{Int}(A.cols)
end

function _structure_arrays(A::SparseMatrixCSC)
  rows = Vector{Int}(undef, nnz(A))
  cols = Vector{Int}(undef, nnz(A))
  _copy_sparse_structure!(A, rows, cols)
  return rows, cols
end

_structure_values(A::SparseMatrixCOO) = A.vals
_structure_values(A::SparseMatrixCSC) = nonzeros(A)

function _stack_columns(MT, src, getter = identity)
  nbatch = length(src)
  nbatch > 0 || throw(ArgumentError("Need at least one column"))
  first_col = getter(first(src))
  out = MT(undef, length(first_col), nbatch)
  copyto!(view(out, :, 1), first_col)
  for j in 2:nbatch
    copyto!(view(out, :, j), getter(src[j]))
  end
  return out
end

function _repeat_column(MT, col, nbatch)
  out = MT(undef, length(col), nbatch)
  for j in 1:nbatch
    copyto!(view(out, :, j), col)
  end
  return out
end

function _stack_batch_bounds(MT, qps)
  return (
    _stack_columns(MT, qps, qp -> qp.meta.x0),
    _stack_columns(MT, qps, qp -> qp.meta.lvar),
    _stack_columns(MT, qps, qp -> qp.meta.uvar),
    _stack_columns(MT, qps, qp -> qp.meta.lcon),
    _stack_columns(MT, qps, qp -> qp.meta.ucon),
  )
end

function _uniform_batch_setup(qps, name, MT; nnzh, islp, validate, model_name)
  nbatch = length(qps)
  @assert nbatch > 0 "Need at least one model"
  validate && _validate_uniform_batch(qps, model_name)
  qp1 = first(qps)
  x0, lvar, uvar, lcon, ucon = _stack_batch_bounds(MT, qps)
  T = eltype(qp1.data.c)
  meta = _batch_meta(T, MT, qp1.meta, nbatch; x0 = x0, lvar = lvar, uvar = uvar, lcon = lcon, ucon = ucon, nnzh = nnzh, islp = islp, name = name)
  c_batch = _stack_columns(MT, qps, qp -> qp.data.c)
  c0_batch = similar(c_batch, T, nbatch)
  copyto!(c0_batch, T[qp.data.c0[] for qp in qps])
  A_nzvals = _stack_columns(MT, qps, qp -> _structure_values(qp.data.A))
  A_rows, A_cols = _structure_arrays(qp1.data.A)
  jac_identity = collect(1:qp1.meta.nnzj)
  jac_rowptr, jac_colidx = _coo_to_csr(Vector{Int}(A_rows), qp1.meta.ncon)
  jac_val_map = Vector{Int}(A_cols)
  jac_op = _build_op(A_nzvals, jac_rowptr, jac_identity, jac_val_map, jac_colidx)
  return (; qp1, meta, c_batch, c0_batch, A_nzvals, A_rows, A_cols, jac_op)
end

function _adapt_uniform_batch(to, meta, c_batch, c0_batch, A_nzvals, jac_op; nnzh, islp)
  c_batch_adapted = Adapt.adapt(to, c_batch)
  c0_batch_adapted = Adapt.adapt(to, c0_batch)
  A_nzvals_adapted = Adapt.adapt(to, A_nzvals)
  jac_op_adapted = Adapt.adapt(to, jac_op)
  meta_adapted = _adapt_batch_meta(to, meta; nnzh = nnzh, islp = islp)
  return meta_adapted, c_batch_adapted, c0_batch_adapted, A_nzvals_adapted, jac_op_adapted
end

function _objrhs_batch_setup(qps, name, MT; validate, model_name)
  validate && _validate_objrhs_batch(qps, model_name)
  nbatch = length(qps)
  @assert nbatch > 0 "Need at least one model"
  qp1 = first(qps)
  x0, lvar, uvar, lcon, ucon = _stack_batch_bounds(MT, qps)
  c = _stack_columns(MT, qps, qp -> qp.data.c)
  return (; qp1, nbatch, x0, lvar, uvar, lcon, ucon, c, name, MT)
end

function _adapt_qpdata(to, data)
  return QPData(
    Adapt.adapt(to, data.A),
    Adapt.adapt(to, data.c),
    Adapt.adapt(to, data.Q);
    lcon = Adapt.adapt(to, data.lcon),
    ucon = Adapt.adapt(to, data.ucon),
    lvar = Adapt.adapt(to, data.lvar),
    uvar = Adapt.adapt(to, data.uvar),
    c0 = data.c0[],
    _v = Adapt.adapt(to, data._v),
  )
end

function _batch_meta(::Type{T}, ::Type{MT}, meta, nbatch; x0, lvar, uvar, lcon, ucon, nnzh = meta.nnzh, islp = meta.islp, name = meta.name) where {T, MT}
  return NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch,
    meta.nvar;
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = meta.ncon,
    lcon = lcon,
    ucon = ucon,
    nnzj = meta.nnzj,
    nnzh = nnzh,
    minimize = meta.minimize,
    islp = islp,
    name = name,
  )
end

function _adapt_batch_meta(to, meta::NLPModels.BatchNLPModelMeta{T}; nnzh = meta.nnzh, islp = meta.islp) where {T}
  x0 = Adapt.adapt(to, meta.x0)
  return _batch_meta(T, typeof(x0), meta, meta.nbatch;
    x0 = x0,
    lvar = Adapt.adapt(to, meta.lvar),
    uvar = Adapt.adapt(to, meta.uvar),
    lcon = Adapt.adapt(to, meta.lcon),
    ucon = Adapt.adapt(to, meta.ucon),
    nnzh = nnzh,
    islp = islp,
  )
end

export ObjRHSBatchQuadraticModel, BatchQuadraticModel
export ObjRHSBatchLinearModel, BatchLinearModel, batch_model
export LPData, LinearModel, QPData, QuadraticModel
export BatchSparseOp, batch_spmv!
export batch_mapreduce!, batch_maximum!, batch_minimum!, batch_sum!
export gather_columns!, gather_entries!, batch_spmv_subset!
export obj_subset!, grad_subset!, cons_subset!, jac_coord_subset!, hess_coord_subset!

include("batch_mapreduce.jl")
include("batch_spmv.jl")
include("operators.jl")
include("models/single.jl")
include("models/uniform.jl")
include("models/obj_rhs.jl")
include("models/linear.jl")

end # module BatchQuadraticModels
