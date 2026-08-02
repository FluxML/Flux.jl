module Train

using LinearAlgebra
using Optimisers: Optimisers
using Functors: fmap, fmapstructure
using ..Flux: Flux
using GPUArrays: GPUArrays

using ProgressLogging: @progress, @withprogress, @logprogress
using EnzymeCore: Duplicated
using ADTypes: AbstractADType, AutoEnzyme, AutoZygote

export setup, train!

# Tuning for `train!(...; gc_interval = :auto)` (see the GC block in `train!`):
# a step longer than `GC_HIDE_S` seconds is assumed compute-bound enough to hide an
# incremental GC (so we collect every step); for cheaper steps we keep the amortized GC cost
# near `GC_OVERHEAD` of training time.
const GC_HIDE_S = 5e-3
const GC_OVERHEAD = 0.02

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

* Manage memory. Runs an incremental garbage collection adaptively.
"""
function train!(loss, adtype::AbstractADType, model, data, opt; cb = nothing,
                caching_allocator::Bool = false, gc_interval::Union{Integer, Symbol} = :auto)
    isnothing(cb) || error("""train! does not support callback functions.
                                For more control use a loop with `gradient` and `update!`.""")
    gc_interval isa Symbol && gc_interval !== :auto &&
        throw(ArgumentError("`gc_interval` must be a non-negative integer or `:auto`, got `:$gc_interval`"))
    Flux.trainmode!(model)
    cache = caching_allocator ? GPUArrays.AllocCache() : nothing

    # Paced GC (fixed `gc_interval` or `:auto`) only helps when the cache is off — with the
    # cache on, a step's buffers are pinned, so a mid-training GC reclaims nothing. State below
    # is for `gc_interval = :auto`, a timing-based adaptive cadence (see the GC block).
    auto = gc_interval === :auto && cache === nothing
    t_step = 0.0    # EMA of step wall-time (seconds)
    t_gc = 0.0      # EMA of one GC's wall-time (seconds)
    interval = 1    # current number of steps between collections
    since = 0       # steps since the last collection

    @withprogress for (i,d) in enumerate(data)
        d_splat = d isa Tuple ? d : (d,)

        t0 = auto ? time_ns() : zero(UInt64)
        # The first step is run without the cache on purpose. On a GPU it triggers cuDNN's
        # convolution-algorithm search, whose one-off probe workspaces are transient but
        # would be pinned by the cache — they are never reused and can be large enough to
        # blow past GPU memory. From the second step on the algorithm is fixed, so the cache
        # only ever sees the real, reusable training buffers.
        if cache === nothing || i == 1
            opt, model = _train_step!(loss, adtype, model, opt, d_splat, i)
        else
            # Reuse the memory allocated during the previous step, see issue #2523.
            GPUArrays.@cached cache begin
                opt, model = _train_step!(loss, adtype, model, opt, d_splat, i)
            end
        end

        # With the caching allocator off, dead GPU buffers are only reclaimed by the GC, which
        # seldom fires on its own here (the `CuArray` wrappers are tiny on the CPU heap), so
        # reserved GPU memory creeps up (issue #2523). A periodic incremental GC bounds that
        # growth at a fraction of the cost of collecting every step.
        #
        # A fixed `gc_interval` collects every N steps. `gc_interval = :auto` instead picks the
        # cadence from wall-clock timing only (no GPU/backend queries, so it works for every
        # backend):
        #   * A compute-bound step (longer than `GC_HIDE_S`) overlaps an incremental GC with its
        #     own GPU work, so the GC is effectively free — collect every step and keep memory
        #     minimal. (Timing a GC in isolation would *overestimate* its cost here, because the
        #     async frees pipeline with the next step; so we deliberately don't use `t_gc`.)
        #   * A cheap step puts the GC on the critical path (the GPU is idle during it, so timing
        #     it in isolation is accurate). Collect only ~every `t_gc/(ε·t_step)` steps to hold
        #     the amortized cost near ε ≈ 2%.
        # An unusually slow step (typically the backend's own memory reclaim under pressure)
        # halves the interval to preempt further such stalls.
        if auto
            dt = (time_ns() - t0) / 1e9
            spike = t_step > 0.0 && dt > 3 * t_step
            t_step = t_step == 0.0 ? dt : 0.9 * t_step + 0.1 * dt
            base = if t_step >= GC_HIDE_S            # compute-bound: GC hides → every step
                1
            elseif t_gc == 0.0 || t_step == 0.0     # not measured yet
                1
            else                                    # cheap step: bound amortized GC cost to ε
                clamp(round(Int, t_gc / (GC_OVERHEAD * t_step)), 1, 4096)
            end
            interval = spike ? max(1, interval ÷ 2) : clamp(interval + 1, 1, base)
            since += 1
            if since >= interval
                g0 = time_ns()
                GC.gc(false)
                gdt = (time_ns() - g0) / 1e9
                t_gc = t_gc == 0.0 ? gdt : 0.9 * t_gc + 0.1 * gdt
                since = 0
            end
        elseif cache === nothing && gc_interval isa Integer && gc_interval > 0 && i % gc_interval == 0
            GC.gc(false)
        end

        @logprogress Base.haslength(data) ? i/length(data) : nothing
    end
    isnothing(cache) || GPUArrays.unsafe_free!(cache)
    return nothing
end

# A single training step, factored out so that `train!` can run it with or without the
# caching allocator without duplicating the body.
function _train_step!(loss, adtype, model, opt, d_splat, i)
    l, gs = Flux.withgradient(m -> loss(m, d_splat...), adtype, model)

    if !isfinite(l)
        throw(DomainError(lazy"Loss is $l on data item $i, stopping training"))
    end

    return _update!(opt, model, gs[1])
end

_update!(opt_state, model, grads) = Optimisers.update!(opt_state, model, grads)

function _update!(opt_state, model::Duplicated, grad)
    opt_state, model2 = Optimisers.update!(opt_state, model.val, grad)
    return opt_state, Duplicated(model2, model.dval)
end


train!(loss, model, data, opt; cb = nothing, caching_allocator::Bool = false, gc_interval::Union{Integer, Symbol} = :auto) =
    train!(loss, AutoZygote(), model, data, opt; cb, caching_allocator, gc_interval)

# This method let you use Optimisers.Descent() without setup, when there is no state
function train!(loss, model, data, rule::Optimisers.AbstractRule; cb = nothing, caching_allocator::Bool = false, gc_interval::Union{Integer, Symbol} = :auto)
    return train!(loss, model, data, _rule_to_state(model, rule); cb, caching_allocator, gc_interval)
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

train!(loss, model::Duplicated, data, opt; cb = nothing, caching_allocator::Bool = false, gc_interval::Union{Integer, Symbol} = :auto) =
    train!(loss, AutoEnzyme(), model, data, opt; cb, caching_allocator, gc_interval)

# This method let you use Optimisers.Descent() without setup, when there is no state
function train!(loss, model::Duplicated, data, rule::Optimisers.AbstractRule; cb=nothing, caching_allocator::Bool = false, gc_interval::Union{Integer, Symbol} = :auto)
    return train!(loss, model, data, _rule_to_state(model, rule); cb, caching_allocator, gc_interval)
end

end # module Train
