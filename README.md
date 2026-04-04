# BatchQuadraticModels.jl

Batched LP/QP models for `QuadraticModels.jl`.

The following convenience constructors are provided, taking as input `Vector{<:QuadraticModel}`:

- `BatchQuadraticModel(qps)`: Uniform-batch model. Sparsity structure is the same but values of A, H may change.
- `BatchLinearModel(qps)`: BatchQuadraticModel where H is zero.
- `ObjRHSBatchQuadraticModel(qps)`: BatchQuadraticModel where A and H are fixed.
- `ObjRHSBatchLinearModel(qps)`: BatchQuadraticModel where A is fixed and H is zero.
- `batch_model(qps)`: Analyze the individual models to construct the most specific batch model possible.
