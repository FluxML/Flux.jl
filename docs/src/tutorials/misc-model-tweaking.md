# Choosing differentiable/gpu parts of the model
!!! note 
    This tutorial features somewhat disconnected topics about customizing your
    models even further. It is advised to be familiar with
    [`Flux.@layer`](@ref), [`Flux.@functor`](@ref), [`freeze!`](@ref
    Flux.freeze!) and other basics of Flux.

Flux provides several ways of freezing, excluding from backprop entirely and
marking custom struct fields not to be moved to the GPU
([Functors.@functor](@ref)) hence excluded from being trained. The following
subsections should make it clear which one suits your needs the best.

## On-the-fly freezing per model instance
Perhaps you'd like to freeze some of the weights of the model (even at
mid-training), and Flux accomplishes this through [`freeze!`](@ref Flux.freeze!) and `thaw!`.

```julia
m = Chain(
      Dense(784 => 64, relu), # freeze this one
      Dense(64 => 64, relu),
      Dense(32 => 10)
    )
opt_state = Flux.setup(Momentum(), m);

# Freeze some layers right away
Flux.freeze!(opt_state.layers[1])

for data in train_set
    input, label = data

    # Some params could be frozen during the training:
    Flux.freeze!(opt_state.layers[2])

    grads = Flux.gradient(m) do m
        result = m(input)
        loss(result, label)
    end
    Flux.update!(opt_state, m, grads[1])

    # Optionally unfreeze the params later
    Flux.thaw!(opt_state.layers[1])
end
```

## Static freezing per model definition
Sometimes some parts of the model ([`Flux.@layer`](@ref)) needn't to be trained at all but these params
still need to reside on the GPU (these params are still needed in the forward
and/or backward pass).
```julia
struct MaskedLayer{T}
    chain::Chain
    mask::T
end
Flux.@layer MyLayer trainable=(chain,)
# mask field will not be updated in the training loop

function (m::MaskedLayer)(x)
    # mask field will still move to to gpu for efficient operations:
  return m.chain(x) + x + m.mask
end

model = MaskedLayer(...) # this model will not have the `mask` field trained
```
Note how this method permanently sets some model fields to be excluded from
training without on-the-fly changing.

## Excluding from model definition
Sometimes some parameters aren't just "not trainable" but they shouldn't even
transfer to the GPU (or be part of the functor). All scalar fields are like this
by default, so things like learning rate multipliers are not trainable nor
transferred to the GPU by default.
```julia
struct CustomLayer{T, F}
    chain::Chain
    activation_results::Vector{F}
    lr_multiplier::Float32
end
Flux.@functor CustomLayer (chain, ) # Explicitly leaving out `activation_results`

function (m::CustomLayer)(x)
    result = m.chain(x) + x
    
    # `activation_results` are not part of the GPU loop, hence we could do
    # things like `push!`
    push!(m.activation_results, mean(result))
    return result
end
```
See more about this in [`Flux.@functor`](@ref)


## Freezing Layer Parameters (deprecated)

When it is desired to not include all the model parameters (for e.g. transfer learning), we can simply not pass in those layers into our call to `params`.

!!! compat "Flux ≤ 0.14"
    The mechanism described here is for Flux's old "implicit" training style.
    When upgrading for Flux 0.15, it should be replaced by [`freeze!`](@ref Flux.freeze!) and `thaw!`.

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

