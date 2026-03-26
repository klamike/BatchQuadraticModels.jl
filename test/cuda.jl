using Adapt
using CUDA
using KernelAbstractions

@testset "CUDA" begin
  if !CUDA.functional()
    @info "Skipping CUDA tests because CUDA.functional() is false"
    return
  end

  @testset "ObjRHSBatchQuadraticModel sparse adaptation" begin
    qps = [ineqconqp_QP() for _ in 1:3]
    cpu_bqp = ObjRHSBatchQuadraticModel(qps)
    gpu_bqp = adapt(CuArray, cpu_bqp)

    @test !(gpu_bqp.data.H isa CuMatrix)
    @test !(gpu_bqp.data.A isa CuMatrix)

    xs = [[1.0, 2.0], [0.5, 1.5], [-0.5, 1.0]]
    ys = [[-1.0, -2.0, 0.5], [-0.5, -1.0, 0.0], [0.0, 0.5, 1.0]]
    bx = CuArray(reduce(hcat, xs))
    by = CuArray(reduce(hcat, ys))
    w = CuArray([1.0, 1.0, 1.0])

    bf = Array(NLPModels.obj(gpu_bqp, bx))
    bg = Array(NLPModels.grad(gpu_bqp, bx))
    bc = Array(NLPModels.cons(gpu_bqp, bx))
    bj = Array(NLPModels.jac_coord(gpu_bqp, bx))
    bh = Array(NLPModels.hess_coord(gpu_bqp, bx, by, w))

    for i in 1:length(qps)
      @test bf[i] ≈ obj(qps[i], xs[i])
      @test bg[:, i] ≈ grad(qps[i], xs[i])
      @test bc[:, i] ≈ cons(qps[i], xs[i])
      @test bj[:, i] ≈ jac_coord(qps[i], xs[i])
      @test bh[:, i] ≈ hess_coord(qps[i], xs[i], ys[i]; obj_weight = Array(w)[i])
    end
  end

  @testset "BatchQuadraticModel adaptation" begin
    qp = ineqconqp_QP()
    Arows = qp.data.A.rows
    Acols = qp.data.A.cols
    Avals = qp.data.A.vals
    Hrows = qp.data.H.rows
    Hcols = qp.data.H.cols
    Hvals = qp.data.H.vals

    models = [
      QuadraticModel(
        qp.data.c .+ shift,
        copy(Hrows), copy(Hcols), Hvals .* scale;
        Arows = copy(Arows),
        Acols = copy(Acols),
        Avals = Avals .+ ashift,
        lcon = qp.meta.lcon,
        ucon = qp.meta.ucon,
        lvar = qp.meta.lvar,
        uvar = qp.meta.uvar,
        c0 = qp.data.c0 + c0shift,
      ) for (shift, scale, ashift, c0shift) in
        ((0.0, 1.0, 0.0, 0.0), (0.2, 2.0, 0.1, 1.0), (-0.1, 0.5, -0.1, -0.5))
    ]

    cpu_bqp = BatchQuadraticModel(models)
    gpu_bqp = adapt(CuArray, cpu_bqp)

    xs = [[1.0, 2.0], [0.5, 1.5], [-0.5, 1.0]]
    ys = [[-1.0, -2.0, 0.5], [-0.5, -1.0, 0.0], [0.0, 0.5, 1.0]]
    bx = CuArray(reduce(hcat, xs))
    by = CuArray(reduce(hcat, ys))
    w = CuArray([1.0, 2.0, 0.5])

    bf = Array(NLPModels.obj(gpu_bqp, bx))
    bg = Array(NLPModels.grad(gpu_bqp, bx))
    bc = Array(NLPModels.cons(gpu_bqp, bx))
    bj = Array(NLPModels.jac_coord(gpu_bqp, bx))
    bh = Array(NLPModels.hess_coord(gpu_bqp, bx, by, w))

    for i in 1:length(models)
      @test bf[i] ≈ obj(models[i], xs[i])
      @test bg[:, i] ≈ grad(models[i], xs[i])
      @test bc[:, i] ≈ cons(models[i], xs[i])
      @test bj[:, i] ≈ jac_coord(models[i], xs[i])
      @test bh[:, i] ≈ hess_coord(models[i], xs[i], ys[i]; obj_weight = Array(w)[i])
    end
  end
end
