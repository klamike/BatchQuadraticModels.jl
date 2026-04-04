@testset "BatchQuadraticModel" begin
  qp = ineqconqp_QP()
  nvar = qp.meta.nvar
  ncon = qp.meta.ncon
  nnzj = qp.meta.nnzj
  nnzh = qp.meta.nnzh
  nbatch = 3

  Arows = qp.data.A.rows
  Acols = qp.data.A.cols
  Avals_base = qp.data.A.vals
  Hrows = qp.data.H.rows
  Hcols = qp.data.H.cols
  Hvals_base = qp.data.H.vals

  c_batches = [copy(qp.data.c), qp.data.c .* 2.0, qp.data.c .+ 0.5]
  c0_batches = [qp.data.c0, 2.0, -1.0]
  Avals_batches = [copy(Avals_base), Avals_base .* 1.5, Avals_base .+ 0.1]
  Hvals_batches = [copy(Hvals_base), Hvals_base .* 3.0, Hvals_base .+ 0.5]
  lvar_batches = [copy(qp.meta.lvar), qp.meta.lvar .- 0.5, qp.meta.lvar .+ 0.5]
  uvar_batches = [copy(qp.meta.uvar), qp.meta.uvar .+ 1.0, qp.meta.uvar .+ 2.0]
  lcon_batches = [copy(qp.meta.lcon), qp.meta.lcon .- 0.1, qp.meta.lcon .+ 0.1]
  ucon_batches = [copy(qp.meta.ucon), qp.meta.ucon .+ 0.2, qp.meta.ucon .+ 0.3]

  models = [
    QuadraticModel(
      c_batches[i],
      copy(Hrows), copy(Hcols), Hvals_batches[i];
      Arows = copy(Arows),
      Acols = copy(Acols),
      Avals = Avals_batches[i],
      lcon = lcon_batches[i],
      ucon = ucon_batches[i],
      lvar = lvar_batches[i],
      uvar = uvar_batches[i],
      c0 = c0_batches[i],
      name = "batch$i",
    ) for i in 1:nbatch
  ]

  bqp = BatchQuadraticModel(models)

  @testset "metadata" begin
    @test bqp.meta.nbatch == nbatch
    @test bqp.meta.nvar == nvar
    @test bqp.meta.ncon == ncon
    @test bqp.meta.nnzj == nnzj
    @test bqp.meta.nnzh == nnzh
    @test size(bqp.meta.x0) == (nvar, nbatch)
    @test size(bqp.meta.lvar) == (nvar, nbatch)
    @test size(bqp.meta.lcon) == (ncon, nbatch)
    for i in 1:nbatch
      @test bqp.meta.lvar[:, i] == lvar_batches[i]
      @test bqp.meta.uvar[:, i] == uvar_batches[i]
      @test bqp.meta.lcon[:, i] == lcon_batches[i]
      @test bqp.meta.ucon[:, i] == ucon_batches[i]
    end
  end

  xs = [[1.0, 2.0], [0.5, 1.5], [-0.5, 1.0]]
  ys = [[-1.0, -2.0, 0.5], [-0.5, -1.0, 0.0], [0.0, 0.5, 1.0]]
  bx = reduce(hcat, xs)
  by = reduce(hcat, ys)
  bobj_weight = [1.0, 2.0, 0.5]

  bf = NLPModels.obj(bqp, bx)
  bg = NLPModels.grad(bqp, bx)
  bc = NLPModels.cons(bqp, bx)
  bjvals = NLPModels.jac_coord(bqp, bx)
  bhvals = NLPModels.hess_coord(bqp, bx, by, bobj_weight)
#   bJv = NLPModels.jprod(bqp, bx, bx)
#   bJtv = NLPModels.jtprod(bqp, bx, by)
#   bHv = NLPModels.hprod(bqp, bx, by, bx, bobj_weight)
  jrows, jcols = NLPModels.jac_structure(bqp)
  hrows, hcols = NLPModels.hess_structure(bqp)

  @testset "batch API matches individual models" begin
    for i in 1:nbatch
      @test bf[i] ≈ obj(models[i], xs[i])
      @test bg[:, i] ≈ grad(models[i], xs[i])
      @test bc[:, i] ≈ cons(models[i], xs[i])
      @test bjvals[:, i] ≈ jac_coord(models[i], xs[i])
      @test bhvals[:, i] ≈ hess_coord(models[i], xs[i], ys[i]; obj_weight = bobj_weight[i])
    #   @test bJv[:, i] ≈ jprod(models[i], xs[i], xs[i])
    #   @test bJtv[:, i] ≈ jtprod(models[i], xs[i], ys[i])
    #   @test bHv[:, i] ≈ hprod(models[i], xs[i], ys[i], xs[i]; obj_weight = bobj_weight[i])

      jrowsi, jcolsi = jac_structure(models[i])
      @test jrows == jrowsi
      @test jcols == jcolsi
      hrowsi, hcolsi = hess_structure(models[i])
      @test hrows == hrowsi
      @test hcols == hcolsi
    end
  end

  @testset "obj_weight scaling" begin
    w = [1.0, 0.0, 2.0]
    bhv = NLPModels.hess_coord(bqp, bx, by, w)
    for i in 1:nbatch
      @test bhv[:, i] ≈ hess_coord(models[i], xs[i], ys[i]; obj_weight = w[i])
    end
  end

  @testset "LP (zero Hessian)" begin
    lp_models = [
      QuadraticModel(
        c_batches[i],
        spzeros(nvar, nvar);
        A = sparse(Arows, Acols, Avals_batches[i], ncon, nvar),
        lcon = lcon_batches[i],
        ucon = ucon_batches[i],
        lvar = lvar_batches[i],
        uvar = uvar_batches[i],
        c0 = c0_batches[i],
      ) for i in 1:nbatch
    ]

    bqp_lp = BatchQuadraticModel(lp_models)
    @test bqp_lp.meta.nnzh == 0

    bf_lp = NLPModels.obj(bqp_lp, bx)
    bg_lp = NLPModels.grad(bqp_lp, bx)
    bc_lp = NLPModels.cons(bqp_lp, bx)
    # bhv_lp = NLPModels.hprod(bqp_lp, bx, by, bx, bobj_weight)

    for i in 1:nbatch
      @test bf_lp[i] ≈ obj(lp_models[i], xs[i])
      @test bg_lp[:, i] ≈ grad(lp_models[i], xs[i])
      @test bc_lp[:, i] ≈ cons(lp_models[i], xs[i])
    #   @test all(bhv_lp[:, i] .== 0.0)
    end
  end

  @testset "unconstrained QP" begin
    unc_models = [
      QuadraticModel(
        c_batches[i],
        copy(Hrows), copy(Hcols), Hvals_batches[i];
        c0 = c0_batches[i],
      ) for i in 1:nbatch
    ]

    bqp_unc = BatchQuadraticModel(unc_models)
    @test bqp_unc.meta.nnzj == 0
    @test bqp_unc.meta.ncon == 0

    bf_unc = NLPModels.obj(bqp_unc, bx)
    bg_unc = NLPModels.grad(bqp_unc, bx)

    for i in 1:nbatch
      @test bf_unc[i] ≈ obj(unc_models[i], xs[i])
      @test bg_unc[:, i] ≈ grad(unc_models[i], xs[i])
    end
  end

  @testset "single batch element" begin
    bqp1 = BatchQuadraticModel([models[1]])
    @test bqp1.meta.nbatch == 1
    bx1 = reshape(xs[1], nvar, 1)
    by1 = reshape(ys[1], ncon, 1)
    @test NLPModels.obj(bqp1, bx1)[1] ≈ obj(models[1], xs[1])
    @test NLPModels.grad(bqp1, bx1)[:, 1] ≈ grad(models[1], xs[1])
    @test NLPModels.cons(bqp1, bx1)[:, 1] ≈ cons(models[1], xs[1])
  end

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
    @test bc_sub ≈ bc[:, roots]
    @test bj_sub ≈ bjvals[:, roots]
    @test bh_sub ≈ bhvals[:, roots]
  end
end

@testset "BatchLinearModel" begin
  qp = QuadraticModel(
    [1.0, -2.0],
    spzeros(2, 2);
    A = sparse([1, 2, 2], [1, 1, 2], [1.0, -1.0, 2.0], 2, 2),
    lcon = [-1.0, 0.0],
    ucon = [2.0, 1.0],
    c0 = -0.5,
    name = "linear_uniform",
  )

  models = [
    QuadraticModel(
      qp.data.c .+ cshift,
      spzeros(2, 2);
      A = sparse([1, 2, 2], [1, 1, 2], avals, 2, 2),
      lcon = qp.meta.lcon .+ lshift,
      ucon = qp.meta.ucon .+ ushift,
      c0 = qp.data.c0 + c0shift,
      name = "lp$i",
    ) for (i, (cshift, avals, lshift, ushift, c0shift)) in enumerate((
      (0.0, [1.0, -1.0, 2.0], 0.0, 0.0, 0.0),
      (0.2, [1.2, -1.1, 2.3], -0.2, 0.1, 0.5),
      (-0.3, [0.8, -0.7, 1.9], 0.1, -0.1, -0.25),
    ))
  ]

  bqp = BatchLinearModel(models)
  xs = [[1.0, 2.0], [0.5, 1.5], [-0.5, 1.0]]
  bx = reduce(hcat, xs)

  bf = NLPModels.obj(bqp, bx)
  bg = NLPModels.grad(bqp, bx)
  bc = NLPModels.cons(bqp, bx)

  @test bqp.meta.islp
  @test bqp.meta.nnzh == 0
  for i in 1:length(models)
    @test bf[i] ≈ obj(models[i], xs[i])
    @test bg[:, i] ≈ grad(models[i], xs[i])
    @test bc[:, i] ≈ cons(models[i], xs[i])
  end
end
