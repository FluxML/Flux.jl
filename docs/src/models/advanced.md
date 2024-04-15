# [Defining Customised Layers](@id man-advanced)

Here we will try and describe usage of some more advanced features that Flux provides to give more control over model building.

## Custom Model Example

Here is a basic example of a custom model. It simply adds the input to the result from the neural network.

```julia
struct CustomModel{T <: Chain} # Parameter to avoid type instability
  chain::T
end

function (m::CustomModel)(x)
  # Arbitrary code can go here, but note that everything will be differentiated.
  # Zygote does not allow some operations, like mutating arrays.

  return m.chain(x) + x
end

# Call @layer to allow for training. Described below in more detail.
Flux.@layer CustomModel
```
Notice that we parameterized the type of the `chain` field. This is necessary for fast Julia code, so that that struct field can be given a concrete type. `Chain`s have a type parameter fully specifying the types of the layers they contain. By using a type parameter, we are freeing Julia to determine the correct concrete type, so that we do not need to specify the full, possibly quite long, type ourselves.

You can then use the model like:

```julia
chain = Chain(Dense(10 => 10))
model = CustomModel(chain)
model(rand(10))
```

For an intro to Flux and automatic differentiation, see this [tutorial](https://fluxml.ai/tutorials/2020/09/15/deep-learning-flux.html).

## Customising Parameter Collection for a Model

Taking reference from our example `Affine` layer from the [basics](@ref man-basics).

By default all the fields in the `Affine` type are collected as its parameters, however, in some cases it may be desired to hold other metadata in our "layers" that may not be needed for training, and are hence supposed to be ignored while the parameters are collected. With Flux, the way to mark some fields of our layer as trainable is through overloading the `trainable` function:

```julia-repl
julia> struct Affine
        W
        b
      end

julia> Affine(in::Int, out::Int) = Affine(randn(out, in), randn(out));

julia> (m::Affine)(x) = m.W * x .+ m.b;

julia> Flux.@layer Affine

julia> a = Affine(Float32[1 2; 3 4; 5 6], Float32[7, 8, 9])
Affine(Float32[1.0 2.0; 3.0 4.0; 5.0 6.0], Float32[7.0, 8.0, 9.0])

julia> Flux.trainable(a) # default behavior
(W = Float32[1.0 2.0; 3.0 4.0; 5.0 6.0], b = Float32[7.0, 8.0, 9.0])

julia> Flux.trainable(a::Affine) = (; W = a.W)  # returns a NamedTuple using the field's name

julia> Flux.trainable(a)
(W = Float32[1.0 2.0; 3.0 4.0; 5.0 6.0],)
```

Only the fields returned by `trainable` will be seen by `Flux.setup` and `Flux.update!` for training. But all fields wil be seen by `gpu` and similar functions, for example:

```julia-repl
julia> a |> f16
Affine(Float16[1.0 2.0; 3.0 4.0; 5.0 6.0], Float16[7.0, 8.0, 9.0])
```

Note that there is no need to overload `trainable` to hide fields which do not contain numerical array (for example, activation functions, or Boolean flags). These are always ignored by training.

The exact same method of `trainable` can also be defined using the macro, for convenience:

```julia
Flux.@layer Affine trainable=(W,)
```

There is a second, more severe, kind of restriction possible. This is not recommended, but is included here for completeness. Calling `Functors.@functor Affine (W,)` means that all no exploration of the model will ever visit the other fields: They will not be moved to the GPU by [`gpu`](@ref), and their precision will not be changed by `f32`. This requires the `struct` to have a corresponding constructor that accepts only `W` as an argument.

## Custom multiple input or output layer

Sometimes a model needs to receive several separate inputs at once or produce several separate outputs at once. In other words, there multiple paths within this high-level layer, each processing a different input or producing a different output. A simple example of this in machine learning literature is the [inception module](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf).

We could have a struct that stores the weights of along each path and implement the joining/splitting in the forward pass function. That would mean a new struct for each different block,
e.g. one would have a `TransformerBlock` struct for a transformer block, and a `ResNetBlock` struct for a ResNet block, each block being composed by smaller sub-blocks. This is often the simplest and cleanest way to implement complex models.

This guide instead will show you how to construct a high-level layer (like [`Chain`](@ref)) that is made of multiple sub-layers for each path.

### Multiple inputs: a custom `Join` layer

Our custom `Join` layer will accept multiple inputs at once, pass each input through a separate path, then combine the results together. Note that this layer can already be constructed using [`Parallel`](@ref), but we will first walk through how do this manually.

We start by defining a new struct, `Join`, that stores the different paths and a combine operation as its fields.
```julia
using Flux
using CUDA

# custom join layer
struct Join{T, F}
  combine::F
  paths::T
end

# allow Join(op, m1, m2, ...) as a constructor
Join(combine, paths...) = Join(combine, paths)
```
Notice again that we parameterized the type of the `combine` and `paths` fields. In addition to the performance considerations of concrete types, this allows either field to be `Vector`s, `Tuple`s, or one of each - we don't need to pay attention to which.

The next step is to use [`Flux.@layer`](@ref) to make our struct behave like a Flux layer. This is important so that calling `Flux.setup` on a `Join` maps over the underlying trainable arrays on each path.
```julia
Flux.@layer Join
```

Finally, we define the forward pass. For `Join`, this means applying each `path` in `paths` to each input array, then using `combine` to merge the results.
```julia
(m::Join)(xs::Tuple) = m.combine(map((f, x) -> f(x), m.paths, xs)...)
(m::Join)(xs...) = m(xs)
```

Lastly, we can test our new layer. Thanks to the proper abstractions in Julia, our layer works on GPU arrays out of the box!
```julia
model = Chain(
              Join(vcat,
                   Chain(Dense(1 => 5, relu), Dense(5 => 1)), # branch 1
                   Dense(1 => 2),                             # branch 2
                   Dense(1 => 1)                              # branch 3
                  ),
              Dense(4 => 1)
             ) |> gpu

xs = map(gpu, (rand(1), rand(1), rand(1)))

model(xs)
# returns a single float vector with one value
```

!!! note
    This `Join` layer is available from the [Fluxperimental.jl](https://github.com/FluxML/Fluxperimental.jl) package.


#### Using `Parallel`

Flux already provides [`Parallel`](@ref) that can offer the same functionality. In this case, `Join` is going to just be syntactic sugar for `Parallel`.
```julia
Join(combine, paths) = Parallel(combine, paths)
Join(combine, paths...) = Join(combine, paths)

# use vararg/tuple version of Parallel forward pass
model = Chain(
              Join(vcat,
                   Chain(Dense(1 => 5, relu), Dense(5 => 1)),
                   Dense(1 => 2),
                   Dense(1 => 1)
                  ),
              Dense(4 => 1)
             ) |> gpu

xs = map(gpu, (rand(1), rand(1), rand(1)))

model(xs)
# returns a single float vector with one value
```

### Multiple outputs: a custom `Split` layer

Our custom `Split` layer will accept a single input, then pass the input through a separate path to produce multiple outputs.

We start by following the same steps as the `Join` layer: define a struct, use [`@layer`](@ref), and define the forward pass.
```julia
using Flux
using CUDA

# custom split layer
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@layer Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)
```

Now we can test to see that our `Split` does indeed produce multiple outputs.
```julia
model = Chain(
              Dense(10 => 5),
              Split(Dense(5 => 1, tanh), Dense(5 => 3, tanh), Dense(5 => 2))
             ) |> gpu

model(gpu(rand(10)))
# returns a tuple with three float vectors
```

A custom loss function for the multiple outputs may look like this:
```julia
using Statistics

# assuming model returns the output of a Split
# x is a single input
# ys is a tuple of outputs
function loss(x, ys, model)
  # rms over all the mse
  ŷs = model(x)
  return sqrt(mean(Flux.mse(y, ŷ) for (y, ŷ) in zip(ys, ŷs)))
end
```

!!! note
    This `Split` layer is available from the [Fluxperimental.jl](https://github.com/FluxML/Fluxperimental.jl) package.

