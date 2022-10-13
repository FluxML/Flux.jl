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
    * `data` should iterate tuples or NamedTuples
    * `opt` should be the result of [`Flux.setup`](@ref).
    * Callback functions are not supported.

For example, with these definitions...
```
data = [(x1, y1), (x2, y2), (x3, y3)];  # each element must be a tuple (or NamedTuple)

loss3(m, x, y) = norm(m(x) .- y)  # the model is the first argument

opt = Flux.setup(Adam(), model)  # explicit setup of optimiser momenta
```
...calling `train!(loss3, model, data, opt)` runs a loop much like this:
```
for d in data
    ∂L∂m = Zygote.gradient(loss3, model, d...)[1]
    Optimisers.update!(opt, model, ∂L∂m)
end
```
Stops with a `DomainError` if the loss is infinite or `NaN` at any point.

Returns a vector containing the value of the loss function at each datapoint.

The built-in loss functions accept 3 arguments, allowing for instance `train!(Flux.Losses.mse, model, data, opt)`.

Callback functions are not supported. But see 3-argument `train!(loss, model, opt)` for an
easy way to construct more complicated training loops.

To change the package used to calculate gradients, use [`Flux.@train_autodiff`](@ref).
"""
function train!(loss, model, data, opt)
  losses = Float32[]
  @withprogress for (i,d) in enumerate(data)
    l, (g, _...) = explicit_withgradient(loss, model, data_splat(d)...)
    isfinite(l) || throw(DomainError("loss function returned $l, stopping training"))
    opt, model = Optimisers.update!(opt, model, g)
    push!(losses, l)
    @logprogress Base.haslength(data) ? i/length(data) : nothing
  end
  return losses  # Not entirely sure returning losses is a good idea
end

data_splat(x::T) where T =  error("""train! expects every d in data be a Tuple or a NamedTuple, got $T
                                   To allow this type, define `Flux.Train.data_splat(x::$T) = (x,)`""")
data_splat(x::Tuple) = x
data_splat(x::NamedTuple) = x
data_splat(x::AbstractArray{<:Number}) = (x,)

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

To change the package used to calculate gradients, use [`Flux.@train_autodiff`](@ref).

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

"""
    Flux.@train_autodiff Tracker
    Flux.@train_autodiff Zygote
    Flux.@train_autodiff Yota
    Flux.@train_autodiff Diffractor

This macro allows the use of `train!` with various automatic differentiation (AD) packages,
instead of the default Zygote.jl.

You should load AD package, and then call this macro with the chosen name.
The macro overwrites a method withing Flux, thus is a global setting, lasting until you re-start Julia.

Only works with [Yota.jl](https://github.com/dfdx/Yota.jl),
[Tracker.jl](https://github.com/FluxML/Tracker.jl) (Flux's old AD),
[Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl) (which is not yet registered),
and with the default [Zygote.jl](https://github.com/FluxML/Zygote.jl).

!!! note
    This is mechanism is experimental! And there are known bugs, in particular Tracker will not automatically switch to training mode for `Dropout` etc.
"""
macro train_autodiff(pkg)
  if pkg == :Diffractor
    return quote
      Diffractor.gradient(sin, 0.0)[1] ≈ 1.0  # ensures an error if not loaded
      function Flux.Train.explicit_withgradient(f, args...)
        y, back = Diffractor.∂⃖¹(f, args...)
        dy1 = Flux.Zygote.sensitivity(y)  # Zygote is loaded, and this gives nice errors
        return (; value = y, gradient = Base.tail(back(dy1)))
      end
    end |> esc
  elseif pkg == :Yota
    return quote
      Yota.grad(sin, 0.0) # [2][1] ≈ 1.0
      function Flux.Train.explicit_withgradient(f, args...)
        value, (_, gradient...) = Yota.grad(f, args...)
        return (; value, gradient)
      end
    end |> esc
  elseif pkg == :Tracker
    return quote
      Tracker.withgradient(sum, [1.0]).val == 1.0  # ensures an error if too-old version
      Flux.Train.explicit_withgradient(f, args...) = Tracker.withgradient(f, args...)
    end |> esc
  elseif pkg == :Zygote
    return quote
      Flux.Train.explicit_withgradient(f, args...) = Flux.Zygote.withgradient(f, args...)
    end |> esc
  else
    throw("@train_autodiff expects one of Tracker, Zygote, Yota, or Diffractor. No other arguments are understood.")
  end
end

end # module
