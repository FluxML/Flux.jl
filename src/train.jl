module Train

using LinearAlgebra
using Optimisers: Optimisers
using Functors: fmap

import ..Flux.Optimise: train!, update!  # during 0.13, we add methods to the old functions

export setup, @train_autodiff

using ProgressLogging: @progress, @withprogress, @logprogress  # TODO add progress logging again
using Zygote: Zygote, Params

"""
    opt = setup(rule, model)

This is a version of `Optimisers.setup`, and is the first step before using `train!`.
It differs from `Optimisers.setup` in that it:
* has one extra check for mutability
* has methods which accept Flux's old optimisers, and convert them.

```jldoctest
julia> model = Dense(2=>1, leakyrelu; init=Flux.ones32);

julia> opt = Flux.setup(Momentum(0.11), model)
(weight = Leaf(Momentum{Float64}(0.11, 0.9), Float32[0.0 0.0]), bias = Leaf(Momentum{Float64}(0.11, 0.9), Float32[0.0]), σ = ())

julia> Flux.train!(model, opt) do m  # 3-arg train!, for one data point (x = [0.2, -0.3], y = [0.4])
         sum(m([0.2, -0.3]) .- [0.4]) * 100
       end
-40.1

julia> model.bias  # was zero, mutated by Flux.train!
1-element Vector{Float32}:
 -0.11

julia> opt  # mutated by Flux.train!
(weight = Leaf(Momentum{Float64}(0.11, 0.9), Float32[0.022 -0.033]), bias = Leaf(Momentum{Float64}(0.11, 0.9), Float32[0.11]), σ = ())
```
"""
function setup(rule::Optimisers.AbstractRule, model)
    state = Optimisers.setup(rule, model)
    fmap(model, exclude = Optimisers.isnumeric) do x
      Optimisers.maywrite(x) || error("""model must be fully mutable for `train!` to work, got `x::$(typeof(x))`.
                                         If `x .+= dx` is in fact ok, define `Optimisers.maywrite(::$(typeof(x))) = true`""")
    end
    state
end

# opt = Flux.setup(Adam(), model); train!(model, opt) do m ... 
setup(model, rule::Optimisers.AbstractRule) = setup(rule, model)

"""
    train!(loss, model, data, opt)

Uses a `loss` function and training `data` to improve the `model`'s parameters
according to a particular optimisation rule `opt`.

!!! note
    This method has significant changes from the one in Flux ≤ 0.13:
    * It now takes the `model` itself, not the result of [`Flux.params`](@ref).
      (This is to move away from Zygote's implicit parameter handling.)
    * Instead of `loss` being a function which typically accepts two arguments
      (the input `x` and expected output `y` from each element of `data`)
      now it should typically accept three, the first of which is the `model` itself.
    * `data` must iterate tuples. Each `d in data` is used as `loss(model, d...)`.
    * `opt` should be the result of [`Flux.setup`](@ref), it will warn you if not.
    * Callback functions are not supported.

For example, with these definitions...
```
data = [(x1, y1), (x2, y2), (x3, y3)];  # each element must be a tuple

loss3(m, x, y) = norm(m(x) .- y)        # the model is the first argument

opt = Flux.setup(Adam(), model)         # explicit setup of optimiser momenta
```
...calling `train!(loss3, model, data, opt)` runs a loop much like this:
```
for d in data
    ∂L∂m = Zygote.gradient(loss3, model, d...)[1]
    Optimisers.update!(opt, model, ∂L∂m)
end
```
You can also write this loop yourself, if you need more flexibility.
Besides the loop, `train!` will:

* Stop with a `DomainError` if the loss is infinite or `NaN` at any point.

* Return a vector containing the value of the loss function at each datapoint.

* Show a progress bar using [`@withprogress`](https://github.com/JuliaLogging/ProgressLogging.jl).

Note that the built-in loss functions accept 3 arguments, allowing for instance
`train!(Flux.Losses.mse, model, data, opt)` instead of defining `loss3` as above.

Note that callback functions are not supported. But arbitrary code can be inserted into the loop.
"""
function train!(loss, model, data, opt)
  Base.issingletontype(typeof(loss)) || error("""train! with explicit parameter expects a pure loss function.
                                                 It must not close over the model, like loss(x,y) = mse(model(x), y). """)
  losses = Float32[]
  @withprogress for (i,d) in enumerate(data)
    d isa Tuple || error("""train! expects as data an iterator producing tuples, but got $(typeof(d)).
                            Pass it `((d,) for d in data)`, or use `gradient` and `update!` for more control.""")
    l, (g, _...) = explicit_withgradient(loss, model, d...)
    isfinite(l) || throw(DomainError("loss function returned $l, stopping training"))
    opt, model = Optimisers.update!(opt, model, g)
    push!(losses, l)
    @logprogress Base.haslength(data) ? i/length(data) : nothing
  end
  return losses  # Not entirely sure returning losses is a good idea
end

"""
    train!(loss, model, opt)
    
Uses a `loss` function improve the `model`'s parameters.

While the 4-argument method of `train!` iterates over a dataset,
this 3-argument method is for a single datapoint, and calls `gradient` just once.
It expects a function `loss` which takes just one argument, the model.
For example:
```
opt = Flux.setup(Adam(), model)   # explicit setup
train!(model, opt) do m           # the model is passed to the function as `m`
    Flux.crossentropy(m(x1), y1)  # but the data point `(x1, y1)` is closed over.
end
```
This calls `Zygote.withgradient(m -> Flux.crossentropy(m(x1), y1), model)`.
(The `do` block is another syntax for this anonymous function.)
Then it updates the parameters contained within `model` according to `opt`.
Finally it returns the value of the loss function.

To iterate over a dataset, writing a loop allows more control than
calling 4-argument `train!`. For example, this adds printing and an early stop:
```
data = Flux.DataLoader((Xtrain, Ytrain), batchsize=32)
opt = Flux.setup(Adam(), model)
for (i, d) in enumerate(data)
    x, y = d
    ell = Flux.train!(m -> Flux.crossentropy(m(x), y), model, opt)
    i%10==0 && println("on step \$i, the loss was \$ell")  # prints every 10th step
    ell<0.1 && break                                     # stops training
end
```

!!! note
    This method has no implicit `Params` analog in Flux ≤ 0.13.
"""
function train!(loss, model, opt)
  l, (g, _...) = explicit_withgradient(loss, model)
  isfinite(l) || return l
  _, model = Optimisers.update!(opt, model, g)
  return l
end

# These methods let you use Optimisers.Descent() without setup, when there is no state
function train!(loss, model, data, rule::Optimisers.AbstractRule)
  train!(loss, model, data, _rule_to_state(model, rule))
end
function train!(loss, model, rule::Optimisers.AbstractRule)
  train!(loss, model, _rule_to_state(model, rule))
end

function _rule_to_state(model, rule::Optimisers.AbstractRule)
  state = setup(rule, model)
  @gensym warn_id
  name = typeof(rule).name.name
  fmap(state, exclude = x -> x isa Optimisers.Leaf) do leaf
    leaf.state isa Nothing ||  @warn """Optimiser $name has state which will be discarded after `train!` finishes.
                                        Please run `opt = Flux.setup($name(), model)` and pass this `opt` to `train!`.""" leaf maxlog=1 _id=warn_id
    leaf
  end
  state
end

explicit_withgradient(f, args...) = Zygote.withgradient(f, args...)  # can overload this to use e.g. Yota / Diffractor

end # module
