using MathOptInterface

const MOI = MathOptInterface

@testset "MOI qp_model conversion" begin
  model = MOI.Utilities.Model{Float64}()
  x = MOI.add_variable(model)
  y = MOI.add_variable(model)
  con = MOI.add_constraint(
    model,
    MOI.ScalarAffineFunction(
      [MOI.ScalarAffineTerm(2.0, x), MOI.ScalarAffineTerm(-1.0, y)],
      1.0,
    ),
    MOI.LessThan(5.0),
  )
  MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
  MOI.set(
    model,
    MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
    MOI.ScalarQuadraticFunction(
      [MOI.ScalarQuadraticTerm(3.0, x, x), MOI.ScalarQuadraticTerm(4.0, x, y)],
      [MOI.ScalarAffineTerm(5.0, x)],
      7.0,
    ),
  )

  qp, index_map = BatchQuadraticModels.qp_model(model)
  z = [2.0, 3.0]

  @test index_map[con].value == 1
  @test NLPModels.obj(qp, z) ≈ MOI.Utilities.eval_variables(vi -> z[index_map[vi].value], MOI.get(
    model,
    MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
  ))
  @test NLPModels.grad(qp, z) ≈ [23.0, 8.0]
  @test qp.data.Q.source[1, 1] ≈ 3.0
  @test qp.data.Q.source[2, 1] ≈ 4.0
  @test qp.meta.ucon == [4.0]
end
