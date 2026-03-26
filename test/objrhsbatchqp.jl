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
      qp.data.H.rows,
      qp.data.H.cols,
      qp.data.H.vals;
      Arows = qp.data.A.rows,
      Acols = qp.data.A.cols,
      Avals = qp.data.A.vals,
      lcon = lcon_batches[i],
      ucon = ucon_batches[i],
      lvar = lvar_batches[i],
      uvar = uvar_batches[i],
      c0 = qp.data.c0,
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
end
