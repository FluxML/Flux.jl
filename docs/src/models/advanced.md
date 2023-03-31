# [Defining Customised Layers](@id man-advanced)

Here we will try and describe usage of some more advanced features that Flux provides to give more control over model building.

## Custom Model Example

Here is a basic example of a custom model. It simply adds the input to the result from the neural network.

```julia
struct CustomModel
  chain::Chain
end

function (m::CustomModel)(x)
  # Arbitrary code can go here, but note that everything will be differentiated.
  # Zygote does not allow some operations, like mutating arrays.

  return m.chain(x) + x
end

# Call @functor to allow for training. Described below in more detail.
Flux.@functor CustomModel
```

You can then use the model like:

```julia
chain = Chain(Dense(10, 10))
model = CustomModel(chain)
model(rand(10))
```

For an intro to Flux and automatic differentiation, see this [tutorial](https://fluxml.ai/tutorials/2020/09/15/deep-learning-flux.html).

## Customising Parameter Collection for a Model

Taking reference from our example `Affine` layer from the [basics](@ref man-basics).

By default all the fields in the `Affine` type are collected as its parameters, however, in some cases it may be desired to hold other metadata in our "layers" that may not be needed for training, and are hence supposed to be ignored while the parameters are collected. With Flux, it is possible to mark the fields of our layers that are trainable in two ways.

The first way of achieving this is through overloading the `trainable` function.

```julia-repl
julia> @functor Affine

julia> a = Affine(rand(3,3), rand(3))
Affine{Array{Float64,2},Array{Float64,1}}([0.66722 0.774872 0.249809; 0.843321 0.403843 0.429232; 0.683525 0.662455 0.065297], [0.42394, 0.0170927, 0.544955])

julia> Flux.params(a) # default behavior
Params([[0.66722 0.774872 0.249809; 0.843321 0.403843 0.429232; 0.683525 0.662455 0.065297], [0.42394, 0.0170927, 0.544955]])

julia> Flux.trainable(a::Affine) = (a.W,)

julia> Flux.params(a)
Params([[0.66722 0.774872 0.249809; 0.843321 0.403843 0.429232; 0.683525 0.662455 0.065297]])
```

Only the fields returned by `trainable` will be collected as trainable parameters of the layer when calling `Flux.params`.

Another way of achieving this is through the `@functor` macro directly. Here, we can mark the fields we are interested in by grouping them in the second argument:

```julia
Flux.@functor Affine (W,)
```

However, doing this requires the `struct` to have a corresponding constructor that accepts those parameters.

## Freezing Layer Parameters

When it is desired to not include all the model parameters (for e.g. transfer learning), we can simply not pass in those layers into our call to `params`.

!!! compat "Flux ≤ 0.13"
    The mechanism described here is for Flux's old "implicit" training style.
    When upgrading for Flux 0.14, it should be replaced by [`freeze!`](@ref Flux.freeze!) and `thaw!`.

Consider a simple multi-layer perceptron model where we want to avoid optimising the first two `Dense` layers. We can obtain
this using the slicing features `Chain` provides:

```julia
m = Chain(
      Dense(784 => 64, relu),
      Dense(64 => 64, relu),
      Dense(32 => 10)
    );

ps = Flux.params(m[3:end])
```

The `Zygote.Params` object `ps` now holds a reference to only the parameters of the layers passed to it.

During training, the gradients will only be computed for (and applied to) the last `Dense` layer, therefore only that would have its parameters changed.

`Flux.params` also takes multiple inputs to make it easy to collect parameters from heterogenous models with a single call. A simple demonstration would be if we wanted to omit optimising the second `Dense` layer in the previous example. It would look something like this:

```julia
Flux.params(m[1], m[3:end])
```

Sometimes, a more fine-tuned control is needed.
We can freeze a specific parameter of a specific layer which already entered a `Params` object `ps`,
by simply deleting it from `ps`:

```julia
ps = Flux.params(m)
delete!(ps, m[2].bias) 
```

## Custom multiple input or output layer

Sometimes a model needs to receive several separate inputs at once or produce several separate outputs at once. In other words, there multiple paths within this high-level layer, each processing a different input or producing a different output. A simple example of this in machine learning literature is the [inception module](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf).

Naively, we could have a struct that stores the weights of along each path and implement the joining/splitting in the forward pass function. But that would mean a new struct any time the operations along each path changes. Instead, this guide will show you how to construct a high-level layer (like [`Chain`](@ref)) that is made of multiple sub-layers for each path.

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
Notice that we parameterized the type of the `paths` field. This is necessary for fast Julia code; in general, `T` might be a `Tuple` or `Vector`, but we don't need to pay attention to what it specifically is. The same goes for the `combine` field.

The next step is to use [`Functors.@functor`](@ref) to make our struct behave like a Flux layer. This is important so that calling `params` on a `Join` returns the underlying weight arrays on each path.
```julia
Flux.@functor Join
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

We start by following the same steps as the `Join` layer: define a struct, use [`Functors.@functor`](@ref), and define the forward pass.
```julia
using Flux
using CUDA

# custom split layer
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

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

