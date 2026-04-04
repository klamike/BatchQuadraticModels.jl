struct BatchLinearModel{T, MT, VT <: AbstractVector{T}, VI <: AbstractVector{Int}} <: AbstractUniformBatchQuadraticModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  c_batch::MT
  c0_batch::VT
  A_nzvals::MT
  A_rows::VI
  A_cols::VI
  jac_op::BatchSparseOp
  jact_op::BatchSparseOp
end

function BatchLinearModel(
  qps::Vector{QP};
  name::String = "SameStructBatchLP",
  validate::Bool = true,
  MT = nothing,
) where {QP <: QuadraticModel{T}} where {T}
  nbatch = length(qps)
  @assert nbatch > 0 "Need at least one model"
  validate && _validate_uniform_batch(qps, "BatchLinearModel")
  qp1 = first(qps)
  MT = _resolve_batch_matrix_type(qp1, T, MT)
  nvar = qp1.meta.nvar
  ncon = qp1.meta.ncon
  nnzj = qp1.meta.nnzj
  @assert qp1.meta.nnzh == 0 "BatchLinearModel requires linear models"

  for qp in qps[2:end]
    @assert qp.meta.nvar == nvar "All models must have same nvar"
    @assert qp.meta.ncon == ncon "All models must have same ncon"
    @assert qp.meta.nnzj == nnzj "All models must have same nnzj"
    @assert qp.meta.nnzh == 0 "All models must be linear"
  end

  x0 = MT(reduce(hcat, [qp.meta.x0 for qp in qps]))
  lvar = MT(reduce(hcat, [qp.meta.lvar for qp in qps]))
  uvar = MT(reduce(hcat, [qp.meta.uvar for qp in qps]))
  lcon = MT(reduce(hcat, [qp.meta.lcon for qp in qps]))
  ucon = MT(reduce(hcat, [qp.meta.ucon for qp in qps]))

  meta = NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch,
    nvar;
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = 0,
    minimize = qp1.meta.minimize,
    islp = true,
    name = name,
  )

  c_batch = MT(undef, nvar, nbatch)
  for (i, qp) in enumerate(qps)
    copyto!(view(c_batch, :, i), qp.data.c)
  end
  c0_batch = similar(c_batch, T, nbatch)
  copyto!(c0_batch, T[qp.data.c0 for qp in qps])

  A_nzvals = MT(undef, nnzj, nbatch)
  for (i, qp) in enumerate(qps)
    copyto!(view(A_nzvals, :, i), nonzeros(qp.data.A))
  end

  A_rows_vec, A_cols_vec = _structure_arrays(qp1.data.A)

  jac_identity = collect(1:nnzj)
  jac_rowptr, jac_colidx = _coo_to_csr(Vector{Int}(A_rows_vec), ncon)
  jac_val_map = Vector{Int}(A_cols_vec)
  jac_op = _build_op(A_nzvals, jac_rowptr, jac_identity, jac_val_map, jac_colidx)

  jact_rowptr, jact_colidx = _coo_to_csr(Vector{Int}(A_cols_vec), nvar)
  jact_val_map = Vector{Int}(A_rows_vec)
  jact_op = _build_op(A_nzvals, jact_rowptr, copy(jac_identity), jact_val_map, jact_colidx)

  return BatchLinearModel{T, MT, typeof(c0_batch), typeof(A_rows_vec)}(
    meta,
    c_batch,
    c0_batch,
    A_nzvals,
    A_rows_vec,
    A_cols_vec,
    jac_op,
    jact_op,
  )
end

function Adapt.adapt_structure(to, bqp::BatchLinearModel{T}) where {T}
  c_batch_adapted = Adapt.adapt(to, bqp.c_batch)
  c0_batch_adapted = Adapt.adapt(to, bqp.c0_batch)
  A_nzvals_adapted = Adapt.adapt(to, bqp.A_nzvals)
  jac_op_adapted = Adapt.adapt(to, bqp.jac_op)
  jact_op_adapted = Adapt.adapt(to, bqp.jact_op)

  MT = typeof(c_batch_adapted)
  meta_adapted = NLPModels.BatchNLPModelMeta{T, MT}(
    bqp.meta.nbatch,
    bqp.meta.nvar;
    x0 = Adapt.adapt(to, bqp.meta.x0),
    lvar = Adapt.adapt(to, bqp.meta.lvar),
    uvar = Adapt.adapt(to, bqp.meta.uvar),
    ncon = bqp.meta.ncon,
    lcon = Adapt.adapt(to, bqp.meta.lcon),
    ucon = Adapt.adapt(to, bqp.meta.ucon),
    nnzj = bqp.meta.nnzj,
    nnzh = 0,
    minimize = bqp.meta.minimize,
    islp = true,
    name = bqp.meta.name,
  )

  return BatchLinearModel{T, MT, typeof(c0_batch_adapted), typeof(bqp.A_rows)}(
    meta_adapted,
    c_batch_adapted,
    c0_batch_adapted,
    A_nzvals_adapted,
    bqp.A_rows,
    bqp.A_cols,
    jac_op_adapted,
    jact_op_adapted,
  )
end

function NLPModels.obj!(bqp::BatchLinearModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  bf_mat = reshape(bf, 1, length(bf))
  batch_mapreduce!(*, +, zero(T), bf_mat, bqp.c_batch, bx)
  bf .+= bqp.c0_batch
  return bf
end

function obj_subset!(
  bqp::BatchLinearModel{T},
  bx::AbstractMatrix,
  bf::AbstractVector,
  roots::AbstractVector{<:Integer},
) where {T}
  HX = similar(bx)
  gather_columns!(HX, bqp.c_batch, roots)
  bf_mat = reshape(bf, 1, length(roots))
  batch_mapreduce!(*, +, zero(T), bf_mat, HX, bx)
  @inbounds for j in eachindex(roots)
    bf[j] += bqp.c0_batch[Int(roots[j])]
  end
  return bf
end

NLPModels.grad!(bqp::BatchLinearModel, bx::AbstractMatrix, bg::AbstractMatrix) = copyto!(bg, bqp.c_batch)

function grad_subset!(
  bqp::BatchLinearModel,
  bx::AbstractMatrix,
  bg::AbstractMatrix,
  roots::AbstractVector{<:Integer},
)
  gather_columns!(bg, bqp.c_batch, roots)
  return bg
end

function NLPModels.cons!(bqp::BatchLinearModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where {T}
  batch_spmv!(bc, bqp.jac_op, bx)
  return bc
end

function cons_subset!(
  bqp::BatchLinearModel{T},
  bx::AbstractMatrix,
  bc::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) where {T}
  batch_spmv_subset!(bc, bqp.jac_op, bx, roots)
  return bc
end

function NLPModels.jac_structure!(
  bqp::BatchLinearModel,
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
)
  @lencheck bqp.meta.nnzj jrows jcols
  copyto!(jrows, bqp.A_rows)
  copyto!(jcols, bqp.A_cols)
  return jrows, jcols
end

function NLPModels.jac_coord!(
  bqp::BatchLinearModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
)
  bjvals .= bqp.A_nzvals
  return bjvals
end

function jac_coord_subset!(
  bqp::BatchLinearModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
  roots::AbstractVector{<:Integer},
)
  gather_columns!(bjvals, bqp.A_nzvals, roots)
  return bjvals
end

function NLPModels.hess_structure!(
  bqp::BatchLinearModel,
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
)
  return hrows, hcols
end

function NLPModels.hess_coord!(
  bqp::BatchLinearModel,
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
)
  return bhvals
end

function hess_coord_subset!(
  bqp::BatchLinearModel,
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
  roots::AbstractVector{<:Integer},
)
  return bhvals
end

struct ObjRHSBatchLinearModel{T, S, M1, M2, MT} <: AbstractObjRHSBatchQuadraticModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  data::QPData{T, S, M1, M2}
  c_batch::MT
end

function ObjRHSBatchLinearModel(
  qp::QuadraticModel{T, S, M1, M2},
  nbatch::Int;
  MT = typeof(similar(qp.data.c, T, 0, 0)),
  x0 = fill!(MT(undef, qp.meta.nvar, nbatch), zero(T)),
  lvar = fill!(MT(undef, qp.meta.nvar, nbatch), T(-Inf)),
  uvar = fill!(MT(undef, qp.meta.nvar, nbatch), T(Inf)),
  lcon = fill!(MT(undef, qp.meta.ncon, nbatch), T(-Inf)),
  ucon = fill!(MT(undef, qp.meta.ncon, nbatch), T(Inf)),
  c = copyto!(MT(undef, qp.meta.nvar, nbatch), repeat(qp.data.c, 1, nbatch)),
  name::String = "ObjRHSBatchLP",
) where {T, S, M1, M2}
  @assert qp.meta.nnzh == 0 "ObjRHSBatchLinearModel requires linear models"
  @assert _supports_objrhs_batch_matrix(qp.data.A) "Dense batch Jacobians are not supported"
  nvar = qp.meta.nvar
  ncon = qp.meta.ncon
  nnzj = qp.meta.nnzj
  meta = NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch,
    nvar;
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = 0,
    minimize = qp.meta.minimize,
    islp = true,
    name = name,
  )
  A_op = sparse_operator(qp.data.A)
  data = QPData(
    qp.data.c0,
    qp.data.c,
    qp.data.v,
    qp.data.H,
    A_op,
    qp.data.regularize,
    qp.data.selected,
    qp.data.σ,
  )
  return ObjRHSBatchLinearModel{T, typeof(data.c), typeof(data.H), typeof(data.A), MT}(meta, data, c)
end

function ObjRHSBatchLinearModel(
  qps::Vector{QP};
  name::String = "ObjRHSBatchLP",
  validate::Bool = true,
  MT = nothing,
) where {QP <: QuadraticModel{T, S, M1, M2}} where {T, S, M1, M2}
  validate && _validate_objrhs_batch(qps, "ObjRHSBatchLinearModel")
  nbatch = length(qps)
  @assert nbatch > 0 "Need at least one model"
  qp1 = first(qps)
  MT = _resolve_batch_matrix_type(qp1, T, MT)
  @assert qp1.meta.nnzh == 0 "ObjRHSBatchLinearModel requires linear models"
  x0 = reduce(hcat, [qp.meta.x0 for qp in qps])
  lvar = reduce(hcat, [qp.meta.lvar for qp in qps])
  uvar = reduce(hcat, [qp.meta.uvar for qp in qps])
  lcon = reduce(hcat, [qp.meta.lcon for qp in qps])
  ucon = reduce(hcat, [qp.meta.ucon for qp in qps])
  c = reduce(hcat, [qp.data.c for qp in qps])
  return ObjRHSBatchLinearModel(
    qp1,
    nbatch;
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    lcon = lcon,
    ucon = ucon,
    c = c,
    name = name,
    MT = MT,
  )
end

function Adapt.adapt_structure(to, bnlp::ObjRHSBatchLinearModel{T}) where {T}
  adapted = _adapt_to_operator(to, bnlp)
  if adapted === nothing
    c_adapted = Adapt.adapt(to, bnlp.data.c)
    v_adapted = Adapt.adapt(to, bnlp.data.v)
    H_adapted = Adapt.adapt(to, bnlp.data.H)
    A_adapted = Adapt.adapt(to, bnlp.data.A)
    data_adapted = QPData(
      bnlp.data.c0,
      c_adapted,
      v_adapted,
      H_adapted,
      A_adapted,
      bnlp.data.regularize,
      bnlp.data.selected,
      bnlp.data.σ,
    )
    c_batch_adapted = Adapt.adapt(to, bnlp.c_batch)
  else
    data_adapted, c_batch_adapted = adapted
  end

  MT = typeof(c_batch_adapted)
  meta_adapted = NLPModels.BatchNLPModelMeta{T, MT}(
    bnlp.meta.nbatch,
    bnlp.meta.nvar;
    x0 = Adapt.adapt(to, bnlp.meta.x0),
    lvar = Adapt.adapt(to, bnlp.meta.lvar),
    uvar = Adapt.adapt(to, bnlp.meta.uvar),
    ncon = bnlp.meta.ncon,
    lcon = Adapt.adapt(to, bnlp.meta.lcon),
    ucon = Adapt.adapt(to, bnlp.meta.ucon),
    nnzj = bnlp.meta.nnzj,
    nnzh = 0,
    minimize = bnlp.meta.minimize,
    islp = true,
    name = bnlp.meta.name,
  )

  return ObjRHSBatchLinearModel{T, typeof(data_adapted.c), typeof(data_adapted.H), typeof(data_adapted.A), MT}(
    meta_adapted, data_adapted, c_batch_adapted,
  )
end

function NLPModels.obj!(bqp::ObjRHSBatchLinearModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  bf_mat = reshape(bf, 1, length(bf))
  batch_mapreduce!(*, +, zero(T), bf_mat, bqp.c_batch, bx)
  bf .+= bqp.data.c0
  return bf
end

function obj_subset!(
  bqp::ObjRHSBatchLinearModel{T},
  bx::AbstractMatrix,
  bf::AbstractVector,
  roots::AbstractVector{<:Integer},
) where {T}
  CX = similar(bx)
  gather_columns!(CX, bqp.c_batch, roots)
  bf_mat = reshape(bf, 1, length(roots))
  batch_mapreduce!(*, +, zero(T), bf_mat, CX, bx)
  bf .+= bqp.data.c0
  return bf
end

NLPModels.grad!(bqp::ObjRHSBatchLinearModel, bx::AbstractMatrix, bg::AbstractMatrix) = copyto!(bg, bqp.c_batch)

function grad_subset!(
  bqp::ObjRHSBatchLinearModel,
  bx::AbstractMatrix,
  bg::AbstractMatrix,
  roots::AbstractVector{<:Integer},
)
  gather_columns!(bg, bqp.c_batch, roots)
  return bg
end

function NLPModels.cons!(bqp::ObjRHSBatchLinearModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where {T}
  mul!(bc, bqp.data.A, bx)
  return bc
end

function cons_subset!(
  bqp::ObjRHSBatchLinearModel{T},
  bx::AbstractMatrix,
  bc::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) where {T}
  mul!(bc, bqp.data.A, bx)
  return bc
end

function NLPModels.jac_structure!(
  bqp::ObjRHSBatchLinearModel,
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
)
  @lencheck bqp.meta.nnzj jrows jcols
  return _copy_sparse_structure!(jrows, jcols, bqp.data.A)
end

function NLPModels.jac_coord!(
  bqp::ObjRHSBatchLinearModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
) 
  return _copy_sparse_values!(bjvals, bqp.data.A)
end

function jac_coord_subset!(
  bqp::ObjRHSBatchLinearModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) 
  return _copy_sparse_values!(bjvals, bqp.data.A)
end

function NLPModels.hess_structure!(
  bqp::ObjRHSBatchLinearModel,
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
)
  return hrows, hcols
end

function NLPModels.hess_coord!(
  bqp::ObjRHSBatchLinearModel,
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
)
  return bhvals
end

function hess_coord_subset!(
  bqp::ObjRHSBatchLinearModel,
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
  roots::AbstractVector{<:Integer},
)
  return bhvals
end

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

function _validate_uniform_batch(qps::Vector{QP}, model_name::AbstractString) where {QP <: QuadraticModel}
  _check_batch_compatibility(qps)
  @assert _shares_uniform_structure(qps) "$model_name requires identical sparse structure across the batch; pass validate=false to skip this check"
  return qps
end

function _validate_objrhs_batch(qps::Vector{QP}, model_name::AbstractString) where {QP <: QuadraticModel}
  _check_batch_compatibility(qps)
  @assert _shares_objrhs_data(qps) "$model_name requires shared static data across the batch; pass validate=false to skip this check"
  return qps
end

_all_linear(qps::Vector{QP}) where {QP <: QuadraticModel} = all(qp -> qp.meta.nnzh == 0, qps)

function _shares_objrhs_data(qps::Vector{QP}) where {QP <: QuadraticModel}
  qp1 = _check_batch_compatibility(qps)
  if _all_linear(qps)
    _supports_objrhs_batch_matrix(qp1.data.A) || return false
    return all(
      qp -> qp.data.c0 == qp1.data.c0 &&
            _same_matrix_values(qp.data.A, qp1.data.A),
      qps[2:end],
    )
  end

  (_supports_objrhs_batch_matrix(qp1.data.H) && _supports_objrhs_batch_matrix(qp1.data.A)) || return false
  return all(
    qp -> qp.data.c0 == qp1.data.c0 &&
          _same_matrix_values(qp.data.H, qp1.data.H) &&
          _same_matrix_values(qp.data.A, qp1.data.A),
    qps[2:end],
  )
end

function _shares_uniform_structure(qps::Vector{QP}) where {QP <: QuadraticModel}
  qp1 = _check_batch_compatibility(qps)
  return all(
    qp -> _same_matrix_structure(qp.data.H, qp1.data.H) &&
          _same_matrix_structure(qp.data.A, qp1.data.A),
    qps[2:end],
  )
end

function batch_model(qps::Vector{QP}; validate::Bool = true, kwargs...) where {QP <: QuadraticModel}
  if _shares_objrhs_data(qps)
    return _all_linear(qps) ? ObjRHSBatchLinearModel(qps; validate = validate, kwargs...) : ObjRHSBatchQuadraticModel(qps; validate = validate, kwargs...)
  end
  if _shares_uniform_structure(qps)
    return _all_linear(qps) ? BatchLinearModel(qps; validate = validate, kwargs...) : BatchQuadraticModel(qps; validate = validate, kwargs...)
  end
  error("Unable to select a batch model: the batch does not share common static data or a common sparsity structure")
end

const LinearBatchQuadraticModel = BatchLinearModel
