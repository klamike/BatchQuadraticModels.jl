using BatchQuadraticModels
using NLPModels
using SparseArrays
using Test

@testset "Batch standard_form matches scalar per-column" begin
  # LP: A is 1×2, VAR_LB_UB, CON_RANGE.
  A = sparse([1.0 1.0])
  Q = sparse([0.0 0.0; 0.0 0.0])
  qp_template = QuadraticModel(QPData(A, [1.0, 2.0], Q;
    lcon = [0.0], ucon = [5.0],
    lvar = [1.0, 1.0], uvar = [3.0, 4.0],
    c0 = 10.0))
  nbatch = 3
  bqp = ObjRHSBatchQuadraticModel(qp_template, nbatch)
  bqp.c_batch .= [1.0  3.0  -2.0; 2.0 -1.0  5.0]
  for j in 1:nbatch
    bqp.meta.lcon[:, j] .= [0.0 + 0.2 * (j - 1)]
    bqp.meta.ucon[:, j] .= [5.0 + 0.2 * (j - 1)]
    bqp.meta.lvar[:, j] .= [1.0 + 0.1 * (j - 1), 1.0 + 0.1 * (j - 1)]
    bqp.meta.uvar[:, j] .= [3.0 + 0.1 * (j - 1), 4.0 + 0.1 * (j - 1)]
    bqp.meta.x0[:, j]   .= [1.5, 2.0]
    bqp.meta.y0[:, j]   .= [0.1 * j]
  end

  std, ws = standard_form(bqp)
  @test size(std.c_batch)    == (NLPModels.get_nvar(std), nbatch)
  @test size(std.meta.lcon)  == (NLPModels.get_ncon(std), nbatch)
  @test length(ws.c0_batch)  == nbatch

  for j in 1:nbatch
    qp_j = QuadraticModel(QPData(
      A, Vector(bqp.c_batch[:, j]), sparse([0.0 0.0; 0.0 0.0]);
      lcon = Vector(bqp.meta.lcon[:, j]), ucon = Vector(bqp.meta.ucon[:, j]),
      lvar = Vector(bqp.meta.lvar[:, j]), uvar = Vector(bqp.meta.uvar[:, j]),
      c0 = 10.0,
    ))
    qp_j.meta.x0 .= bqp.meta.x0[:, j]
    qp_j.meta.y0 .= bqp.meta.y0[:, j]
    std_j, ws_j = standard_form(qp_j)

    @test Vector(std.c_batch[:, j])   ≈ std_j.data.c
    @test Vector(std.meta.lcon[:, j]) ≈ std_j.data.lcon
    @test Vector(std.meta.x0[:, j])   ≈ std_j.meta.x0
    @test Vector(std.meta.y0[:, j])   ≈ std_j.meta.y0
    @test ws.c0_batch[j]              ≈ std_j.data.c0[]
  end
end

@testset "Batch update_standard_form! selective updates" begin
  A = sparse([1.0 1.0])
  Q = sparse([0.0 0.0; 0.0 0.0])
  qp_template = QuadraticModel(QPData(A, [1.0, 2.0], Q;
    lcon = [0.0], ucon = [5.0],
    lvar = [1.0, 1.0], uvar = [3.0, 4.0], c0 = 10.0))
  nbatch = 2
  bqp = ObjRHSBatchQuadraticModel(qp_template, nbatch)
  bqp.c_batch .= [1.0 3.0; 2.0 -1.0]
  for j in 1:nbatch
    bqp.meta.lcon[:, j] .= [0.0]
    bqp.meta.ucon[:, j] .= [5.0]
    bqp.meta.lvar[:, j] .= [1.0, 1.0]
    bqp.meta.uvar[:, j] .= [3.0, 4.0]
    bqp.meta.x0[:, j]   .= [1.5, 2.0]
  end
  std, ws = standard_form(bqp)

  # Selective c update.
  new_c = [5.0 7.0; 5.0 -2.0]
  update_standard_form!(bqp, std, ws; c = new_c)
  bqp_ref = ObjRHSBatchQuadraticModel(qp_template, nbatch)
  bqp_ref.c_batch .= new_c
  for j in 1:nbatch
    bqp_ref.meta.lcon[:, j] .= [0.0]
    bqp_ref.meta.ucon[:, j] .= [5.0]
    bqp_ref.meta.lvar[:, j] .= [1.0, 1.0]
    bqp_ref.meta.uvar[:, j] .= [3.0, 4.0]
    bqp_ref.meta.x0[:, j]   .= [1.5, 2.0]
  end
  std_ref, ws_ref = standard_form(bqp_ref)
  @test std.c_batch    == std_ref.c_batch
  @test std.meta.lcon  == std_ref.meta.lcon
  @test ws.c0_batch    == ws_ref.c0_batch

  # Selective lvar update.
  new_lvar = [1.5 2.0; 1.5 2.0]
  update_standard_form!(bqp, std, ws; lvar = new_lvar)
  bqp_ref.meta.lvar .= new_lvar
  std_ref, ws_ref = standard_form(bqp_ref)
  @test std.c_batch    ≈ std_ref.c_batch
  @test std.meta.lcon  ≈ std_ref.meta.lcon
  @test ws.c0_batch    ≈ ws_ref.c0_batch
end

@testset "Batch recover_primal! inverts VAR_LB_UB" begin
  A = sparse([1.0 1.0])
  Q = sparse([0.0 0.0; 0.0 0.0])
  qp_template = QuadraticModel(QPData(A, [1.0, 2.0], Q;
    lcon = [0.0], ucon = [5.0],
    lvar = [1.0, 1.0], uvar = [3.0, 4.0], c0 = 0.0))
  nbatch = 2
  bqp = ObjRHSBatchQuadraticModel(qp_template, nbatch)
  for j in 1:nbatch
    bqp.meta.lcon[:, j] .= [0.0]
    bqp.meta.ucon[:, j] .= [5.0]
    bqp.meta.lvar[:, j] .= [1.0, 1.0]
    bqp.meta.uvar[:, j] .= [3.0, 4.0]
    bqp.meta.x0[:, j]   .= [1.5, 2.0]
  end
  std, ws = standard_form(bqp)
  # Build a valid std primal z > 0, satisfying z[idx1] + z[idx2] == uvar - lvar.
  nstd_var = NLPModels.get_nvar(std)
  z = ones(nstd_var, nbatch)
  x = similar(bqp.meta.lvar)
  recover_primal!(x, ws, z)
  # For VAR_LB_UB with lvar=1, uvar=3: x = lvar + z[idx1] = 1 + 1 = 2.
  @test x[1, 1] ≈ 2.0
  @test x[2, 1] ≈ 2.0  # lvar=1, uvar=4 → x = lvar + 1 = 2
end
