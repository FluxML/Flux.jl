# [Flat vs. Nested Structures](@id man-destructure)


A Flux model is a nested structure, with parameters stored within many layers. Sometimes you may want a flat representation of them, to interact with functions expecting just one vector. This is provided by `destructure`:

```julia
julia> model = Chain(Dense(2=>1, tanh), Dense(1=>1))
Chain(
  Dense(2 => 1, tanh),                  # 3 parameters
  Dense(1 => 1),                        # 2 parameters
)                   # Total: 4 arrays, 5 parameters, 276 bytes.

julia> flat, rebuild = Flux.destructure(model)
(Float32[0.863101, 1.2454957, 0.0, -1.6345707, 0.0], Restructure(Chain, ..., 5))

julia> rebuild(zeros(5))  # same structure, new parameters
Chain(
  Dense(2 => 1, tanh),                  # 3 parameters  (all zero)
  Dense(1 => 1),                        # 2 parameters  (all zero)
)                   # Total: 4 arrays, 5 parameters, 276 bytes.
```

Both `destructure` and the `Restructure` function can be used within gradient computations. For instance, this computes the Hessian `∂²L/∂θᵢ∂θⱼ` of some loss function, with respect to all parameters of the Flux model. The resulting matrix has off-diagonal entries, which cannot really be expressed in a nested structure:

```julia
julia> x = rand(Float32, 2, 16);

julia> grad = gradient(m -> sum(abs2, m(x)), model)  # nested gradient
((layers = ((weight = Float32[10.339018 11.379145], bias = Float32[22.845667], σ = nothing), (weight = Float32[-29.565302;;], bias = Float32[-37.644184], σ = nothing)),),)

julia> function loss(v::Vector)
         m = rebuild(v)
         y = m(x)
         sum(abs2, y)
       end;

julia> gradient(loss, flat)  # flat gradient, same numbers
(Float32[10.339018, 11.379145, 22.845667, -29.565302, -37.644184],)

julia> Zygote.hessian(loss, flat)  # second derivative
5×5 Matrix{Float32}:
  -7.13131   -5.54714  -11.1393  -12.6504   -8.13492
  -5.54714   -7.11092  -11.0208  -13.9231   -9.36316
 -11.1393   -11.0208   -13.7126  -27.9531  -22.741
 -12.6504   -13.9231   -27.9531   18.0875   23.03
  -8.13492   -9.36316  -22.741    23.03     32.0

julia> Flux.destructure(grad)  # acts on non-models, too
(Float32[10.339018, 11.379145, 22.845667, -29.565302, -37.644184], Restructure(Tuple, ..., 5))
```

!!! compat "Flux ≤ 0.12"
    Old versions of Flux had an entirely different implementation of `destructure`, which
    had many bugs (and almost no tests). Many comments online still refer to that now-deleted
    function, or to memories of it.


### All Parameters

The function `destructure` now lives in [`Optimisers.jl`](https://github.com/FluxML/Optimisers.jl).
(Be warned this package is unrelated to the `Flux.Optimisers` sub-module! The confusion is temporary.)

```@docs
Optimisers.destructure
Optimisers.trainable
Optimisers.isnumeric
```

### All Layers

Another kind of flat view of a nested model is provided by the `modules` command. This extracts a list of all layers:

```@docs
Flux.modules
```

### Save and Load

```@docs
Flux.state
Flux.loadmodel!
```