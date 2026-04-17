function _parse_variables(model)
  vars = MOI.get(model, MOI.ListOfVariableIndices())
  nvar = length(vars)
  lvar = zeros(nvar)
  uvar = zeros(nvar)
  x0 = zeros(nvar)
  has_start = MOI.VariablePrimalStart() in MOI.get(model, MOI.ListOfVariableAttributesSet())

  index_map = MOI.Utilities.IndexMap()
  for (i, vi) in enumerate(vars)
    index_map[vi] = MOI.VariableIndex(i)
  end

  for (i, vi) in enumerate(vars)
    lvar[i], uvar[i] = MOI.Utilities.get_bounds(model, Float64, vi)
    if has_start
      value = MOI.get(model, MOI.VariablePrimalStart(), vi)
      value !== nothing && (x0[i] = value)
    end
  end

  return index_map, nvar, lvar, uvar, x0
end

function _parse_constraints(moimodel, index_map)
  _index(v::MOI.VariableIndex) = index_map[v].value

  nlin = 0
  linrows = Int[]
  lincols = Int[]
  linvals = Float64[]
  lin_lcon = Float64[]
  lin_ucon = Float64[]

  contypes = MOI.get(moimodel, MOI.ListOfConstraintTypesPresent())
  for (F, S) in contypes
    @assert F <: AF || F <: VI
    conindices = MOI.get(moimodel, MOI.ListOfConstraintIndices{F, S}())
    for cidx in conindices
      fun = MOI.get(moimodel, MOI.ConstraintFunction(), cidx)
      if F == VI
        index_map[cidx] = MOI.ConstraintIndex{F, S}(_index(fun))
        continue
      else
        index_map[cidx] = MOI.ConstraintIndex{F, S}(nlin)
      end
      set = MOI.get(moimodel, MOI.ConstraintSet(), cidx)
      if F <: SAF
        for term in fun.terms
          push!(linrows, nlin + 1)
          push!(lincols, _index(term.variable))
          push!(linvals, term.coefficient)
        end
        if set isa Union{MOI.Interval{Float64}, MOI.GreaterThan{Float64}}
          push!(lin_lcon, -fun.constant + set.lower)
        elseif set isa MOI.EqualTo{Float64}
          push!(lin_lcon, -fun.constant + set.value)
        else
          push!(lin_lcon, -Inf)
        end
        if set isa Union{MOI.Interval{Float64}, MOI.LessThan{Float64}}
          push!(lin_ucon, -fun.constant + set.upper)
        elseif set isa MOI.EqualTo{Float64}
          push!(lin_ucon, -fun.constant + set.value)
        else
          push!(lin_ucon, Inf)
        end
        nlin += 1
      end
    end
  end
  return linrows, lincols, linvals, lin_lcon, lin_ucon
end

function _parse_objective(moimodel, index_map, nvar)
  _index(v::MOI.VariableIndex) = index_map[v].value

  constant = 0.0
  vect = zeros(Float64, nvar)
  rows = Int[]
  cols = Int[]
  vals = Float64[]

  fobj = MOI.get(moimodel, MOI.ObjectiveFunction{LinQuad}())

  if fobj isa VI
    vect[_index(fobj)] = 1.0
  elseif fobj isa SAF
    constant = fobj.constant
    for term in fobj.terms
      vect[_index(term.variable)] += term.coefficient
    end
  elseif fobj isa SQF
    MOI.Utilities.canonicalize!(fobj)
    constant = fobj.constant
    for term in fobj.affine_terms
      vect[_index(term.variable)] += term.coefficient
    end
    for term in fobj.quadratic_terms
      i = _index(term.variable_1)
      j = _index(term.variable_2)
      push!(rows, max(i, j))
      push!(cols, min(i, j))
      push!(vals, term.coefficient)
    end
  end
  return rows, cols, vals, vect, constant
end

function qp_model(moimodel::MOI.ModelLike)
  index_map, nvar, lvar, uvar, x0 = _parse_variables(moimodel)
  nvar == 0 && throw(ArgumentError("Trivial MOI models with no decision variables are not supported."))
  Ai, Aj, Ax, lb, ub = _parse_constraints(moimodel, index_map)
  Qi, Qj, Qx, c, c0 = _parse_objective(moimodel, index_map, nvar)

  minimize = MOI.get(moimodel, MOI.ObjectiveSense()) == MOI.MIN_SENSE
  ncon = length(lb)
  A = sparse(Ai, Aj, Ax, ncon, nvar)
  Q = sparse(Qi, Qj, Qx, nvar, nvar)
  data = QPData(
    A,
    c,
    Q;
    lcon = lb,
    ucon = ub,
    lvar = lvar,
    uvar = uvar,
    c0 = c0,
  )
  return QuadraticModel(data; x0 = x0, minimize = minimize), index_map
end
