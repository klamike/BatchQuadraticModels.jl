@testset "ObjRHSBatchQuadraticModel" begin
  qp = ineqconqp_QP()
  nvar = qp.meta.nvar
  ncon = qp.meta.ncon
  nnzj = qp.meta.nnzj
  nnzh = qp.meta.nnzh
  nbatch = 3

  lvar_batches = [
    copy(qp.meta.lvar),
    qp.meta.lvar .- 0.5,
    qp.meta.lvar .+ 0.5,
  ]
  uvar_batches = [
    copy(qp.meta.uvar),
    qp.meta.uvar .+ 1.0,
    qp.meta.uvar .+ 2.0,
  ]
  lcon_batches = [
    copy(qp.meta.lcon),
    qp.meta.lcon .- 0.1,
    qp.meta.lcon .+ 0.1,
  ]
  ucon_batches = [
    copy(qp.meta.ucon),
    qp.meta.ucon .+ 0.2,
    qp.meta.ucon .+ 0.3,
  ]
  c_batches = [
    copy(qp.data.c),
    qp.data.c .+ 0.1,
    qp.data.c .- 0.2,
  ]

  models = [
    QuadraticModel(
      c_batches[i],
      qp.data.Q.source.rows,
      qp.data.Q.source.cols,
      qp.data.Q.source.vals;
      Arows = qp.data.A.source.rows,
      Acols = qp.data.A.source.cols,
      Avals = qp.data.A.source.vals,
      lcon = lcon_batches[i],
      ucon = ucon_batches[i],
      lvar = lvar_batches[i],
      uvar = uvar_batches[i],
      c0 = qp.data.c0[],
      name = "batch$i",
    ) for i in 1:nbatch
  ]

  blvar = reduce(hcat, lvar_batches)
  buvar = reduce(hcat, uvar_batches)
  blcon = reduce(hcat, lcon_batches)
  bucon = reduce(hcat, ucon_batches)
  bc = reduce(hcat, c_batches)
  bx0 = reduce(hcat, [models[i].meta.x0 for i in 1:nbatch])

  bqp = ObjRHSBatchQuadraticModel(
    qp,
    nbatch;
    x0 = bx0,
    lvar = blvar,
    uvar = buvar,
    lcon = blcon,
    ucon = bucon,
    c = bc,
    name = "ObjRHSBatchQP",
  )

  @test bqp.meta.nbatch == nbatch
  @test bqp.meta.nvar == nvar
  @test bqp.meta.ncon == ncon
  @test bqp.c_batch == bc
  @test bqp.A isa BatchQuadraticModels.AbstractSparseOperator
  @test bqp.Q isa BatchQuadraticModels.AbstractSparseOperator

  xs = [[1.0, 2.0], [0.5, 1.5], [-0.5, 1.0]]
  ys = [[-1.0, -2.0, 0.5], [-0.5, -1.0, 0.0], [0.0, 0.5, 1.0]]
  bx = reduce(hcat, xs)
  by = reduce(hcat, ys)
  bobj_weight = [1.0, 1.0, 1.0]

  bf = NLPModels.obj(bqp, bx)
  bg = NLPModels.grad(bqp, bx)
  bc_cons = NLPModels.cons(bqp, bx)
  bjvals = NLPModels.jac_coord(bqp, bx)
  bhvals = NLPModels.hess_coord(bqp, bx, by, bobj_weight)
#   bJv = NLPModels.jprod(bqp, bx, bx)
#   bJtv = NLPModels.jtprod(bqp, bx, by)
#   bHv = NLPModels.hprod(bqp, bx, by, bx, bobj_weight)
  jrows, jcols = NLPModels.jac_structure(bqp)
  hrows, hcols = NLPModels.hess_structure(bqp)

  for i in 1:nbatch
    @test bf[i] ≈ obj(models[i], xs[i])
    @test bg[:, i] ≈ grad(models[i], xs[i])
    @test bc_cons[:, i] ≈ cons(models[i], xs[i])
    @test bjvals[:, i] ≈ jac_coord(models[i], xs[i])
    @test bhvals[:, i] ≈ hess_coord(models[i], xs[i], ys[i]; obj_weight = bobj_weight[i])
    # @test bJv[:, i] ≈ jprod(models[i], xs[i], xs[i])
    # @test bJtv[:, i] ≈ jtprod(models[i], xs[i], ys[i])
    # @test bHv[:, i] ≈ hprod(models[i], xs[i], ys[i], xs[i]; obj_weight = bobj_weight[i])

    jrowsi, jcolsi = jac_structure(models[i])
    @test jrows == jrowsi
    @test jcols == jcolsi
    hrowsi, hcolsi = hess_structure(models[i])
    @test hrows == hrowsi
    @test hcols == hcolsi
  end

  bqp2 = ObjRHSBatchQuadraticModel(models; name = "ObjRHSBatchQP_from_vector")
  @test bqp2.meta.nbatch == nbatch
  @test bqp2.meta.nvar == nvar
  @test bqp2.meta.ncon == ncon
  @test bqp2.meta.x0 == bx0
  @test bqp2.meta.lvar == blvar
  @test bqp2.meta.uvar == buvar
  @test bqp2.meta.lcon == blcon
  @test bqp2.meta.ucon == bucon
  @test bqp2.c_batch == bc
  @test NLPModels.obj(bqp2, bx) ≈ NLPModels.obj(bqp, bx)
  @test NLPModels.cons(bqp2, bx) ≈ NLPModels.cons(bqp, bx)

  @testset "subset kernels match selected columns" begin
    roots = Int32[1, 3]
    bx_sub = bx[:, roots]
    by_sub = by[:, roots]
    w_sub = bobj_weight[roots]

    bf_sub = zeros(eltype(bx), length(roots))
    bg_sub = zeros(eltype(bx), nvar, length(roots))
    bc_sub = zeros(eltype(bx), ncon, length(roots))
    bj_sub = zeros(eltype(bx), nnzj, length(roots))
    bh_sub = zeros(eltype(bx), nnzh, length(roots))

    BatchQuadraticModels.obj_subset!(bqp, bx_sub, bf_sub, roots)
    BatchQuadraticModels.grad_subset!(bqp, bx_sub, bg_sub, roots)
    BatchQuadraticModels.cons_subset!(bqp, bx_sub, bc_sub, roots)
    BatchQuadraticModels.jac_coord_subset!(bqp, bx_sub, bj_sub, roots)
    BatchQuadraticModels.hess_coord_subset!(bqp, bx_sub, by_sub, w_sub, bh_sub, roots)

    @test bf_sub ≈ bf[roots]
    @test bg_sub ≈ bg[:, roots]
    @test bc_sub ≈ bc_cons[:, roots]
    @test bj_sub ≈ bjvals[:, roots]
    @test bh_sub ≈ bhvals[:, roots]
  end
end

@testset "ObjRHSBatchQuadraticModel" begin
  qp = QuadraticModel(
    [-1.0, 2.0],
    spzeros(2, 2);
    A = sparse([1, 1, 2], [1, 2, 2], [1.0, -1.0, 2.0], 2, 2),
    lcon = [0.0, -1.0],
    ucon = [1.0, 2.0],
    c0 = 0.5,
    name = "linear_objrhs",
  )
  qps = [
    QuadraticModel(
      qp.data.c .+ shift,
      spzeros(2, 2);
      A = qp.data.A,
      lcon = qp.meta.lcon .+ lshift,
      ucon = qp.meta.ucon .+ ushift,
      c0 = qp.data.c0[],
      name = "lp$i",
    ) for (i, (shift, lshift, ushift)) in enumerate(((0.0, 0.0, 0.0), (0.2, -0.1, 0.1), (-0.3, 0.2, 0.0)))
  ]

  bqp = ObjRHSBatchQuadraticModel(qps)
  xs = [[1.0, 2.0], [0.5, 1.5], [-0.5, 1.0]]
  bx = reduce(hcat, xs)

  bf = NLPModels.obj(bqp, bx)
  bg = NLPModels.grad(bqp, bx)
  bc = NLPModels.cons(bqp, bx)

  @test bqp.meta.islp
  @test bqp.meta.nnzh == 0
  @test bqp.A isa BatchQuadraticModels.AbstractSparseOperator
  for i in 1:length(qps)
    @test bf[i] ≈ obj(qps[i], xs[i])
    @test bg[:, i] ≈ grad(qps[i], xs[i])
    @test bc[:, i] ≈ cons(qps[i], xs[i])
  end
end

@testset "ObjRHS constructor validation" begin
  bounds = (lcon = [-Inf, -Inf], ucon = [Inf, Inf])
  A1 = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
  A2 = sparse([1, 1], [1, 2], [3.0, 4.0], 2, 2)
  lp1 = QuadraticModel([1.0, 2.0], spzeros(2, 2); A = A1, c0 = 0.0, bounds...)
  lp2 = QuadraticModel([3.0, 4.0], spzeros(2, 2); A = A2, c0 = 1.0, bounds...)

  @test_throws AssertionError ObjRHSBatchQuadraticModel([lp1, lp2]; validate = true)
  @test ObjRHSBatchQuadraticModel([lp1, lp2]; validate = false) isa ObjRHSBatchQuadraticModel

  qp = ineqconqp_QP()
  qp_bad = QuadraticModel(
    qp.data.c,
    qp.data.Q.source.rows,
    qp.data.Q.source.cols,
    qp.data.Q.source.vals .* 2.0;
    Arows = qp.data.A.source.rows,
    Acols = qp.data.A.source.cols,
    Avals = qp.data.A.source.vals,
    lcon = qp.meta.lcon,
    ucon = qp.meta.ucon,
    lvar = qp.meta.lvar,
    uvar = qp.meta.uvar,
    c0 = qp.data.c0[],
  )

  @test_throws AssertionError ObjRHSBatchQuadraticModel([qp, qp_bad]; validate = true)
  @test ObjRHSBatchQuadraticModel([qp, qp_bad]; validate = false) isa ObjRHSBatchQuadraticModel
end

@testset "ObjRHS adapt preserves objective sense" begin
  bounds = (lcon = [-Inf, -Inf], ucon = [Inf, Inf])
  A = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)
  lp = QuadraticModel([1.0, 2.0], spzeros(2, 2); A = A, c0 = 0.0, minimize = false, bounds...)
  blp = ObjRHSBatchQuadraticModel([lp, lp])
  @test !blp.meta.minimize
  @test !Adapt.adapt(Array, blp).meta.minimize

  qp = ineqconqp_QP()
  qmax = QuadraticModel(
    qp.data.c,
    qp.data.Q.source.rows,
    qp.data.Q.source.cols,
    qp.data.Q.source.vals;
    Arows = qp.data.A.source.rows,
    Acols = qp.data.A.source.cols,
    Avals = qp.data.A.source.vals,
    lcon = qp.meta.lcon,
    ucon = qp.meta.ucon,
    lvar = qp.meta.lvar,
    uvar = qp.meta.uvar,
    c0 = qp.data.c0[],
    minimize = false,
  )
  bqp = ObjRHSBatchQuadraticModel([qmax, qmax])
  @test !bqp.meta.minimize
  @test !Adapt.adapt(Array, bqp).meta.minimize
end

@testset "ObjRHSBatchQuadraticModel handles LPs" begin
  bounds = (lcon = [-Inf, -Inf], ucon = [Inf, Inf])
  A = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)
  lp = QuadraticModel([1.0, 2.0], spzeros(2, 2); A = A, c0 = 0.5, bounds...)

  bqp = ObjRHSBatchQuadraticModel([lp, lp])
  bx = hcat([1.0, 2.0], [0.5, 1.5])

  @test bqp.meta.islp
  @test bqp.meta.nnzh == 0
  @test NLPModels.obj(bqp, bx) ≈ [obj(lp, bx[:, 1]), obj(lp, bx[:, 2])]
  @test Adapt.adapt(Array, bqp).meta.islp
end

@testset "ObjRHS empty batch validation" begin
  qp = ineqconqp_QP()
  lp = QuadraticModel(
    [1.0, 2.0],
    spzeros(2, 2);
    A = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2),
    lcon = [-Inf, -Inf],
    ucon = [Inf, Inf],
  )

  @test_throws AssertionError ObjRHSBatchQuadraticModel(typeof(qp)[])
  @test_throws AssertionError ObjRHSBatchQuadraticModel(typeof(lp)[])
end
