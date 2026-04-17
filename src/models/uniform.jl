struct BatchQuadraticModel{T, MT, VT <: AbstractVector{T}, VI <: AbstractVector{Int}, JO <: BatchSparseOp, HO <: BatchSparseOp} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  c_batch::MT
  c0_batch::VT
  H_nzvals::MT
  A_nzvals::MT
  hess_rows::VI
  hess_cols::VI
  A_rows::VI
  A_cols::VI
  jac_op::JO
  hess_op::HO
  _HX::MT
end

function BatchQuadraticModel(
  qps::Vector{QP};
  name::String = "SameStructBatchQP",
  validate::Bool = false,
  MT = nothing,
) where {QP <: QuadraticModel{T}} where {T}
  qp1 = _check_batch_compatibility(qps)
  nbatch = length(qps)
  MT = _resolve_batch_matrix_type(qp1, T, MT)
  nvar = qp1.meta.nvar
  nnzh = qp1.meta.nnzh

  uniform = _uniform_batch_setup(qps, name, MT; nnzh = nnzh, islp = qp1.meta.islp, validate = validate, model_name = "BatchQuadraticModel")

  hess_rows_vec, hess_cols_vec = _structure_arrays(qp1.data.Q)
  hess_rows = Vector{Int}(hess_rows_vec)
  hess_cols = Vector{Int}(hess_cols_vec)

  A_nzvals = uniform.A_nzvals
  H_nzvals = _stack_columns(MT, qps, qp -> _structure_values(qp.data.Q))

  off_diag = findall(hess_rows .!= hess_cols)
  sym_scatter_rows = vcat(Vector{Int}(hess_rows), Vector{Int}(hess_cols[off_diag]))
  base_idx = collect(1:nnzh)
  sym_nz_idx = vcat(base_idx, Vector{Int}(off_diag))
  sym_gather_cols = vcat(Vector{Int}(hess_cols), Vector{Int}(hess_rows[off_diag]))

  hess_rowptr, hess_colidx = _coo_to_csr(sym_scatter_rows, nvar)
  hess_op = _build_op(H_nzvals, hess_rowptr, sym_nz_idx, sym_gather_cols, hess_colidx)

  VT = typeof(uniform.c0_batch)
  VI = typeof(hess_rows)
  _HX = fill!(MT(undef, nvar, nbatch), zero(T))

  return BatchQuadraticModel{T, MT, VT, VI, typeof(uniform.jac_op), typeof(hess_op)}(
    uniform.meta,
    uniform.c_batch,
    uniform.c0_batch,
    H_nzvals,
    A_nzvals,
    hess_rows,
    hess_cols,
    uniform.A_rows,
    uniform.A_cols,
    uniform.jac_op,
    hess_op,
    _HX,
  )
end

function Adapt.adapt_structure(to, bqp::BatchQuadraticModel{T}) where {T}
  meta_adapted, c_batch_adapted, c0_batch_adapted, A_nzvals_adapted, jac_op_adapted =
    _adapt_uniform_batch(to, bqp.meta, bqp.c_batch, bqp.c0_batch, bqp.A_nzvals, bqp.jac_op; nnzh = bqp.meta.nnzh, islp = bqp.meta.islp)
  H_nzvals_adapted = Adapt.adapt(to, bqp.H_nzvals)
  hess_op_adapted = Adapt.adapt(to, bqp.hess_op)
  HX_adapted = Adapt.adapt(to, bqp._HX)

  MT = typeof(c_batch_adapted)
  VT = typeof(c0_batch_adapted)

  return BatchQuadraticModel{T, MT, VT, typeof(bqp.hess_rows), typeof(jac_op_adapted), typeof(hess_op_adapted)}(
    meta_adapted,
    c_batch_adapted,
    c0_batch_adapted,
    H_nzvals_adapted,
    A_nzvals_adapted,
    bqp.hess_rows,
    bqp.hess_cols,
    bqp.A_rows,
    bqp.A_cols,
    jac_op_adapted,
    hess_op_adapted,
    HX_adapted,
  )
end

function NLPModels.obj!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  batch_spmv!(bqp._HX, bqp.hess_op, bx)
  bf_mat = reshape(bf, 1, length(bf))
  bqp._HX .*= T(0.5)
  bqp._HX .+= bqp.c_batch
  batch_mapreduce!(*, +, zero(T), bf_mat, bqp._HX, bx)
  bf .+= bqp.c0_batch
  return bf
end

function obj_subset!(
  bqp::BatchQuadraticModel{T},
  bx::AbstractMatrix,
  bf::AbstractVector,
  roots::AbstractVector{<:Integer},
) where {T}
  na = length(roots)
  HX = view(bqp._HX, :, 1:na)
  bf_mat = reshape(bf, 1, na)
  gather_columns!(HX, bqp.c_batch, roots)
  batch_spmv_subset!(HX, bqp.hess_op, bx, roots, T(0.5), one(T))
  batch_mapreduce!(*, +, zero(T), bf_mat, HX, bx)
  @inbounds for j in eachindex(roots)
    bf[j] += bqp.c0_batch[Int(roots[j])]
  end
  return bf
end

function NLPModels.grad!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where {T}
  batch_spmv!(bg, bqp.hess_op, bx)
  bg .+= bqp.c_batch
  return bg
end

function grad_subset!(
  bqp::BatchQuadraticModel{T},
  bx::AbstractMatrix,
  bg::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) where {T}
  gather_columns!(bg, bqp.c_batch, roots)
  batch_spmv_subset!(bg, bqp.hess_op, bx, roots, one(T), one(T))
  return bg
end

function NLPModels.cons!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where {T}
  batch_spmv!(bc, bqp.jac_op, bx)
  return bc
end

function cons_subset!(
  bqp::BatchQuadraticModel{T},
  bx::AbstractMatrix,
  bc::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) where {T}
  batch_spmv_subset!(bc, bqp.jac_op, bx, roots)
  return bc
end

function NLPModels.jac_structure!(
  bqp::BatchQuadraticModel,
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
)
  @lencheck bqp.meta.nnzj jrows jcols
  copyto!(jrows, bqp.A_rows)
  copyto!(jcols, bqp.A_cols)
  return jrows, jcols
end

function NLPModels.jac_coord!(
  bqp::BatchQuadraticModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
)
  bjvals .= bqp.A_nzvals
  return bjvals
end

function jac_coord_subset!(
  bqp::BatchQuadraticModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
  roots::AbstractVector{<:Integer},
)
  gather_columns!(bjvals, bqp.A_nzvals, roots)
  return bjvals
end

function NLPModels.hess_structure!(
  bqp::BatchQuadraticModel,
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
)
  copyto!(hrows, bqp.hess_rows)
  copyto!(hcols, bqp.hess_cols)
  return hrows, hcols
end

function NLPModels.hess_coord!(
  bqp::BatchQuadraticModel{T},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) where {T}
  bhvals .= bqp.H_nzvals .* bobj_weight'
  return bhvals
end

function hess_coord_subset!(
  bqp::BatchQuadraticModel{T},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
  roots::AbstractVector{<:Integer},
) where {T}
  gather_columns!(bhvals, bqp.H_nzvals, roots)
  bhvals .*= bobj_weight'
  return bhvals
end
