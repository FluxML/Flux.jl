# GPU Support

NVIDIA GPU support should work out of the box on systems with CUDA and CUDNN installed. For more details see the [CuArrays](https://github.com/JuliaGPU/CuArrays.jl) readme.

## GPU Usage

Support for array operations on other hardware backends, like GPUs, is provided by external packages like [CuArrays](https://github.com/JuliaGPU/CuArrays.jl). Flux is agnostic to array types, so we simply need to move model weights and data to the GPU and Flux will handle it.

For example, we can use `CuArrays` (with the `cu` converter) to run our [basic example](models/basics.md) on an NVIDIA GPU.

(Note that you need to have CUDA available to use CuArrays – please see the [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl) instructions for more details.)

```julia
using CuArrays

W = cu(rand(2, 5)) # a 2×5 CuArray
b = cu(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = cu(rand(5)), cu(rand(2)) # Dummy data
loss(x, y) # ~ 3
```

Note that we convert both the parameters (`W`, `b`) and the data set (`x`, `y`) to cuda arrays. Taking derivatives and training works exactly as before.

If you define a structured model, like a `Dense` layer or `Chain`, you just need to convert the internal parameters. Flux provides `fmap`, which allows you to alter all parameters of a model at once.

```julia
d = Dense(10, 5, σ)
d = fmap(cu, d)
d.W # CuArray
d(cu(rand(10))) # CuArray output

m = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
m = fmap(cu, m)
d(cu(rand(10)))
```

However, if you create a customized model, `fmap` may not work out of the box.

```julia
julia> struct ActorCritic{A, C}
           actor::A
           critic::C
       end

julia> m = ActorCritic(ones(2,2), ones(2))
ActorCritic{Array{Float64,2},Array{Float64,1}}([1.0 1.0; 1.0 1.0], [1.0, 1.0])

julia> fmap(cu, m)
ActorCritic{Array{Float64,2},Array{Float64,1}}([1.0 1.0; 1.0 1.0], [1.0, 1.0])
```

As you can see, nothing changed after `fmap(cu, m)`. The reason is that `Flux` doesn't know your customized model structure. To make it work as expected, you need the `@functor` macro.

```julia
julia> Flux.@functor ActorCritic

julia> fmap(cu, m)
ActorCritic{CuArray{Float32,2,Nothing},CuArray{Float32,1,Nothing}}(Float32[1.0 1.0; 1.0 1.0], Float32[1.0, 1.0])
```

Now you can see that the inner fields of `actor` and `critic` are transformed into `CuArray`. So what does the `@functor` macro do here? Basically, it will create a function like this:

```julia
Flux.functor(m::ActorCritic) = (actor = m.actor, critic=m.critic), fields -> ActorCritic(fields...)
```

And the `functor` will be called recursively in `fmap`. As you can see, the result of `functor` contains two parts, a *destructure* part and a *reconstrucutre* part. The first part is to make the customized model structure into `trainable` data structure known to `Flux` (here is a `NamedTuple`). The goal is to turn `m` into `(actor=cu(ones(2,2)), critic=cu(ones(2)))`. The second part is to turn the result back into a `ActorCritic`, so that we can get `ActorCritic(cu(ones(2,2)),cu(ones(2)))`.

By default, the `@functor` macro will transform all the fields in your customized structure. In some cases, you may only want to transform several fields. Then you just specify those fields manually like `Flux.@functor ActorCritic (actor,)` (note that the fields part must be a tuple). And make sure the `ActorCritic(actor)` constructor is also implemented.

As a convenience, Flux provides the `gpu` function to convert models and data to the GPU if one is available. By default, it'll do nothing, but loading `CuArrays` will cause it to move data to the GPU instead.

```julia
julia> using Flux, CuArrays

julia> m = Dense(10,5) |> gpu
Dense(10, 5)

julia> x = rand(10) |> gpu
10-element CuArray{Float32,1}:
 0.800225
 ⋮
 0.511655

julia> m(x)
5-element CuArray{Float32,1}:
 -0.30535
 ⋮
 -0.618002
```

The analogue `cpu` is also available for moving models and data back off of the GPU.

```julia
julia> x = rand(10) |> gpu
10-element CuArray{Float32,1}:
 0.235164
 ⋮
 0.192538

julia> x |> cpu
10-element Array{Float32,1}:
 0.235164
 ⋮
 0.192538
```