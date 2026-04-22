# BatchQuadraticModels.jl

`BatchQuadraticModels.jl` provides lightweight LP/QP model types plus batched model wrappers.

- `LPData`, `QPData`, `LinearModel`, `QuadraticModel` are the single-problem containers used by `MadIPM`.
- `BatchQuadraticModel`, `BatchQuadraticModel`, `ObjRHSBatchQuadraticModel`, and `ObjRHSBatchQuadraticModel` cover common batched solve patterns.
- The package also provides CUDA adaptation for the supported sparse/model types and an MOI extension that imports LPs/QPs into `QuadraticModel`.

Typical usage:

```julia
using SparseArrays
using BatchQuadraticModels

data = QPData(
    sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2),
    [1.0, 1.0],
    sparse(Int[], Int[], Float64[], 2, 2);
    lcon = [1.0],
    ucon = [1.0],
    lvar = [0.0, 0.0],
)

qp = QuadraticModel(data)
```
