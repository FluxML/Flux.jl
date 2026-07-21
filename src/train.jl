module Train

using LinearAlgebra
using Optimisers: Optimisers
using Functors: fmap, fmapstructure
using ..Flux: Flux
using GPUArrays: GPUArrays

using ProgressLogging: @progress, @withprogress, @logprogress
using EnzymeCore: Duplicated
using ADTypes: AbstractADType, AutoEnzyme, AutoZygote

export setup, train!, train_step!, TrainState

"""
    opt_state = setup(rule, model)

This is a version of `Optimisers.setup`, and is the first step before using [`train!`](@ref Flux.train!).
It differs from `Optimisers.setup` in that it:
* has one extra check for mutability (since Flux expects to mutate the model in-place,
  while Optimisers.jl is designed to return an updated model)
* has methods which accept Flux's old optimisers, and convert them.
  (The old `Flux.Optimise.Adam` and new `Optimisers.Adam` are distinct types.)

# Example
```jldoctest
julia> model = Dense(2 => 1, leakyrelu; init=ones);

julia> opt_state = Flux.setup(Momentum(0.1), model)  # this encodes the optimiser and its state
(weight = Leaf(Momentum(eta=0.1, rho=0.9), [0.0 0.0]), bias = Leaf(Momentum(eta=0.1, rho=0.9), [0.0]), σ = ())

julia> x1, y1 = [0.2, -0.3], [0.4];  # use the same data for two steps:

julia> Flux.train!(model, [(x1, y1), (x1, y1)], opt_state) do m, x, y
         sum(abs.(m(x) .- y)) * 100
       end

julia> model.bias  # was zero, mutated by Flux.train!
1-element Vector{Float64}:
 10.19

julia> opt_state  # mutated by Flux.train!
(weight = Leaf(Momentum(eta=0.1, rho=0.9), [-2.018 3.027]), bias = Leaf(Momentum(eta=0.1, rho=0.9), [-10.09]), σ = ())
```
"""
function setup(rule::Optimisers.AbstractRule, model)
    state = Optimisers.setup(rule, model)
    # This check only needs foreach; using fmap caused https://github.com/FluxML/Flux.jl/issues/2144
    fmapstructure(model, exclude = Optimisers.isnumeric) do x
        Optimisers.maywrite(x) || error("""model must be fully mutable for `train!` to work, got `x::$(typeof(x))`.
                                         If `x .+= dx` is in fact ok, define `Optimisers.maywrite(::$(typeof(x))) = true`""")
    end
    return state
end

"""
    opt_state = setup(rule, model::Duplicated) = setup(rule, model.val)

Special method for use with Enzyme.jl, ignores the stored gradient.
"""
setup(rule::Optimisers.AbstractRule, model::Duplicated) = setup(rule, model.val)

"""
    train!(loss, [adtype,] model, data, opt_state)

Uses a `loss` function and training `data` to improve the `model`'s parameters
according to a particular optimisation rule encoded in `opt_state`.

Iterates through `data` once, evaluating for each `d in data` either
`loss(model, d...)` if `d isa Tuple`, or else `loss(model, d)` for other `d`.

The optional argument `adtype`, selects an automatic differentiation engine  among the ones supported by 
[`gradient`](@ref). If no `adtype` is given, then Zygote is used by default, unless `model` is of type `Duplicated` from Enzyme.jl,
in which case Enzyme is used.

For example, with these definitions...
```
data = [(x1, y1), (x2, y2), (x3, y3)]

loss3(m, x, y) = norm(m(x) .- y)        # the model is the first argument

opt_state = Flux.setup(Adam(), model)   # explicit setup of optimiser momenta
```
...calling `Flux.train!(loss3, model, data, opt_state)` runs a loop much like this:
```
for d in data
    ∂L∂m = gradient(loss3, model, d...)[1]
    update!(opt_state, model, ∂L∂m)
end
```
You can also write this loop yourself, if you need more flexibility.
For this reason `train!` is not highly extensible.
It adds only a few features to the loop above:

* Stop with a `DomainError` if the loss is infinite or `NaN` at any point.

* Show a progress bar using [`@withprogress`](https://github.com/JuliaLogging/ProgressLogging.jl).

!!! compat "New"
    This method was added in Flux 0.13.9.
    It has significant changes from the one used by Flux ≤ 0.13:
    * It now takes the `model` itself, not the result of `Flux.params`.
      (This is to move away from Zygote's "implicit" parameter handling, with `Grads`.)
    * Instead of `loss` being a function which accepts only the data,
      now it must also accept the `model` itself, as the first argument.
    * `opt_state` should be the result of [`Flux.setup`](@ref). Using an optimiser
      such as `Adam()` without this step should give you a warning.
    * Callback functions are not supported.
      (But any code can be included in the above `for` loop.)
"""
function train!(loss, adtype::AbstractADType, model, data, opt; cb = nothing)
    isnothing(cb) || error("""train! does not support callback functions.
                                For more control use a loop with `gradient` and `update!`.""")
    cache = GPUArrays.AllocCache()
    @withprogress for (i,d) in enumerate(data)
        d_splat = d isa Tuple ? d : (d,)

        GPUArrays.@cached cache begin
            l, gs = Flux.withgradient(m -> loss(m, d_splat...), adtype, model)

            if !isfinite(l)
                throw(DomainError(lazy"Loss is $l on data item $i, stopping training"))
            end

            opt, model = _update!(opt, model, gs[1])
        end

        @logprogress Base.haslength(data) ? i/length(data) : nothing
    end
    GPUArrays.unsafe_free!(cache)
end

_update!(opt_state, model, grads) = Optimisers.update!(opt_state, model, grads)

function _update!(opt_state, model::Duplicated, grad)
    opt_state, model2 = Optimisers.update!(opt_state, model.val, grad)
    return opt_state, Duplicated(model2, model.dval)
end


train!(loss, model, data, opt; cb = nothing) = train!(loss, AutoZygote(), model, data, opt; cb)

# This method let you use Optimisers.Descent() without setup, when there is no state
function train!(loss, model, data, rule::Optimisers.AbstractRule; cb = nothing)
    return train!(loss, model, data, _rule_to_state(model, rule); cb)
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
    return state
end

train!(loss, model::Duplicated, data, opt; cb = nothing) = train!(loss, AutoEnzyme(), model, data, opt; cb)

# This method let you use Optimisers.Descent() without setup, when there is no state
function train!(loss, model::Duplicated, data, rule::Optimisers.AbstractRule; cb=nothing)
    return train!(loss, model, data, _rule_to_state(model, rule); cb)
end

"""
    TrainState(rule, model)

Container bundling everything needed to take training steps with [`train_step!`](@ref Flux.train_step!):
the `model`, the optimisation `rule`, the optimiser state, the number of steps taken so far,
an allocator cache (see [`GPUArrays.AllocCache`](https://juliagpu.github.io/GPUArrays.jl/))
reused across steps to reduce GPU memory pressure, and an optional cache for AD frameworks
that support reusing gradient-preparation state across steps (e.g. Mooncake).

It is the recommended state object when training step-by-step,
in particular when the step is to be compiled (e.g. with Reactant) since the same state is
threaded through every iteration.

The `model` is stored inside the state and is mutated in place by `train_step!`.
For use with Enzyme.jl, pass a `Duplicated` model, exactly as for [`gradient`](@ref Flux.gradient).

# Fields
- `model`: the model being trained.
- `rule`: the `Optimisers.AbstractRule` (e.g. `Adam()`) encoding the optimisation algorithm.
- `opt_state`: the optimiser state, as returned by [`Flux.setup`](@ref).
- `step::Int`: number of calls to `train_step!` made so far.
- `alloc_cache`: a `GPUArrays.AllocCache` used to cache memory allocations across steps.
- `ad_cache`: cache for AD frameworks that support reusing preparation state across steps
  (e.g. Mooncake). `nothing` when unused.

# Example
```julia
model = Chain(Dense(2 => 3, relu), Dense(3 => 1))
state = Flux.TrainState(Adam(1e-3), model)

loss(m, x, y) = Flux.mse(m(x), y)

for data in dataloader
    l = Flux.train_step!(loss, data, state)
    @info "loss" l
end
```

See also [`train_step!`](@ref Flux.train_step!) and [`train!`](@ref Flux.train!).
"""
mutable struct TrainState{TM, TR<:Optimisers.AbstractRule, TS}
    model::TM
    rule::TR
    opt_state::TS
    step::Int
    alloc_cache::GPUArrays.AllocCache
    ad_cache::Any  # untyped so AD backends can stash a freshly-prepared cache across steps
end

function TrainState(rule::Optimisers.AbstractRule, model)
    opt_state = setup(rule, model)
    return TrainState(model, rule, opt_state, 0, GPUArrays.AllocCache(), nothing)
end

function Base.show(io::IO, ts::TrainState)
    print(io, "TrainState(", ts.rule, ", model; step = ", ts.step, ")")
end

"""
    train_step!(loss, [adtype,] data, state::TrainState)

Take a single optimisation step, updating `state.model` in place: compute the gradient of
`loss` with respect to the model and apply the optimisation rule stored in `state`.
The loss value is returned.

Unlike [`train!`](@ref Flux.train!) which iterates over a
whole data iterator, `train_step!` performs exactly one update, leaving the loop to the
caller. This gives full control, and lets the step be compiled (e.g. with Reactant) and
called repeatedly with the same `state`.

The loss is evaluated as `loss(model, data...)` if `data isa Tuple`, otherwise as
`loss(model, data)`, matching [`train!`](@ref Flux.train!).

The optional `adtype` selects the automatic differentiation backend, among those supported
by [`gradient`](@ref Flux.gradient). If omitted, Zygote is used, unless `state.model` is a
`Duplicated` from Enzyme.jl, in which case Enzyme is used.

A `DomainError` is thrown if the loss is infinite or `NaN`.

# Auxiliary information

As with [`withgradient`](@ref Flux.withgradient), the `loss` function may return auxiliary
information alongside the scalar loss, by returning a `Tuple` or `NamedTuple` whose **first**
element is the scalar loss used for the gradient. In that case `train_step!` returns the
whole object, so any extra outputs are available to the caller:

```julia
loss(m, x, y) = (loss = Flux.mse(m(x), y), pred = m(x))  # loss first, then aux

res = Flux.train_step!(loss, (x, y), state)
res.loss   # scalar used for the optimisation step
res.pred   # auxiliary output
```

When `loss` returns just a scalar, that scalar is returned (no wrapping).

!!! note
    Returning auxiliary information is currently only supported with the Zygote backend,
    matching [`withgradient`](@ref Flux.withgradient).

# Example
```julia
model = Chain(Dense(2 => 3, relu), Dense(3 => 1))
state = Flux.TrainState(Adam(1e-3), model)

loss(m, x, y) = Flux.mse(m(x), y)

for epoch in 1:epochs
    for data in dataloader
        Flux.train_step!(loss, data, state)
    end
end
```

See also [`TrainState`](@ref Flux.TrainState) and [`train!`](@ref Flux.train!).
"""
function train_step!(loss, adtype::AbstractADType, data, state::TrainState)
    d_splat = data isa Tuple ? data : (data,)
    val = GPUArrays.@cached state.alloc_cache begin
        val, gs = Flux.withgradient(m -> loss(m, d_splat...), adtype, state.model)
        l = _loss_value(val)
        if !isfinite(l)
            throw(DomainError(lazy"Loss is $l on step $(state.step + 1), stopping training"))
        end
        state.opt_state, state.model = _update!(state.opt_state, state.model, gs[1])
        val
    end
    state.step += 1
    return val
end

# Extract the scalar loss from the value returned by the loss function, which may carry
# auxiliary information as the trailing elements of a Tuple or NamedTuple (loss first).
_loss_value(l) = l
_loss_value(l::Union{Tuple, NamedTuple}) = first(l)

train_step!(loss, data, state::TrainState) = train_step!(loss, AutoZygote(), data, state)

train_step!(loss, data, state::TrainState{<:Duplicated}) = train_step!(loss, AutoEnzyme(), data, state)

end # module Train
