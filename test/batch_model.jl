@testset "batch_model selection" begin
  qp = ineqconqp_QP()
  qps_objrhs = [
    QuadraticModel(
      qp.data.c .+ shift,
      qp.data.Q.rows,
      qp.data.Q.cols,
      qp.data.Q.vals;
      Arows = qp.data.A.rows,
      Acols = qp.data.A.cols,
      Avals = qp.data.A.vals,
      lcon = qp.meta.lcon,
      ucon = qp.meta.ucon,
      c0 = qp.data.c0[],
      name = "qp_objrhs_$i",
    ) for (i, shift) in enumerate((0.0, 0.2, -0.1))
  ]
  @test batch_model(qps_objrhs) isa ObjRHSBatchQuadraticModel

  qps_uniform = [
    QuadraticModel(
      qp.data.c .+ shift,
      qp.data.Q.rows,
      qp.data.Q.cols,
      qp.data.Q.vals .* hscale;
      Arows = qp.data.A.rows,
      Acols = qp.data.A.cols,
      Avals = qp.data.A.vals .+ ashift,
      lcon = qp.meta.lcon,
      ucon = qp.meta.ucon,
      c0 = qp.data.c0[] + c0shift,
      name = "qp_uniform_$i",
    ) for (i, (shift, hscale, ashift, c0shift)) in enumerate(((0.0, 1.0, 0.0, 0.0), (0.2, 2.0, 0.1, 0.5), (-0.1, 0.5, -0.2, -0.25)))
  ]
  @test batch_model(qps_uniform) isa BatchQuadraticModel

  lp = QuadraticModel(
    [1.0, -2.0],
    spzeros(2, 2);
    A = sparse([1, 1, 2], [1, 2, 2], [1.0, -1.0, 2.0], 2, 2),
    lcon = [0.0, -1.0],
    ucon = [1.0, 2.0],
    c0 = 0.5,
    name = "lp",
  )
  lps_objrhs = [
    QuadraticModel(
      lp.data.c .+ shift,
      spzeros(2, 2);
      A = lp.data.A,
      lcon = lp.meta.lcon .+ lshift,
      ucon = lp.meta.ucon .+ ushift,
      c0 = lp.data.c0[],
      name = "lp_objrhs_$i",
    ) for (i, (shift, lshift, ushift)) in enumerate(((0.0, 0.0, 0.0), (0.2, -0.1, 0.1), (-0.3, 0.2, 0.0)))
  ]
  @test batch_model(lps_objrhs) isa ObjRHSBatchLinearModel

  lps_uniform = [
    QuadraticModel(
      lp.data.c .+ shift,
      spzeros(2, 2);
      A = sparse([1, 1, 2], [1, 2, 2], avals, 2, 2),
      lcon = lp.meta.lcon,
      ucon = lp.meta.ucon,
      c0 = lp.data.c0[] + c0shift,
      name = "lp_uniform_$i",
    ) for (i, (shift, avals, c0shift)) in enumerate((
      (0.0, [1.0, -1.0, 2.0], 0.0),
      (0.1, [1.2, -0.8, 2.3], 0.2),
      (-0.2, [0.7, -1.1, 1.8], -0.1),
    ))
  ]
  @test batch_model(lps_uniform) isa BatchLinearModel
end
