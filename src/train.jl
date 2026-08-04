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

# Implemented by FluxReactantExt when Reactant.jl is loaded. Only reached from the ReactantDevice
# branch of `train!` below, which requires the model to hold Reactant arrays — i.e. Reactant
# (and thus the extension) is already loaded. The bare declaration lets `train!` reference it;
# a MethodError is the (essentially unreachable) safety net if the extension is missing.
function _reactant_train! end

# ---------------------------------------------------------------------------
# Paced garbage collection for `train!`.
#
# With the caching allocator off, dead GPU buffers are only reclaimed by the GC, which seldom
# fires on its own here (the `CuArray` wrappers are tiny on the CPU heap), so reserved GPU
# memory creeps up (issue #2523). A periodic incremental GC bounds that growth at a fraction
# of the cost of collecting every step. A `GCPacer` decides, once per training step, whether
# to run `GC.gc(false)`. The `train!` loop brackets each step with `tic`/`maybe_gc!`, keeping
# all the cadence bookkeeping out of the loop body.
# ---------------------------------------------------------------------------
abstract type GCPacer end

# Timestamp the pacer needs at the start of a step; `0` when it doesn't time steps.
tic(::GCPacer) = zero(UInt64)

# No paced GC: with the caching allocator on a mid-step GC reclaims nothing (the step's
# buffers are pinned), and `gc_interval = 0` disables pacing explicitly.
struct NoGCPacer <: GCPacer end
maybe_gc!(::NoGCPacer, i::Integer, t0::UInt64) = nothing

# Fixed cadence: collect every `interval` steps.
struct FixedGCPacer <: GCPacer
    interval::Int
end
function maybe_gc!(p::FixedGCPacer, i::Integer, ::UInt64)
    i % p.interval == 0 && GC.gc(false)
    return nothing
end

# Tuning for the adaptive pacer: a step longer than `GC_HIDE_S` seconds is assumed
# compute-bound enough to hide an incremental GC (so we collect every step); for cheaper steps
# we keep the amortized GC cost near `GC_OVERHEAD` of training time.
const GC_HIDE_S = 5e-3
const GC_OVERHEAD = 0.02

# Adaptive cadence driven by wall-clock timing only (no GPU/backend queries, so it works for
# every backend):
#   * A compute-bound step (longer than `GC_HIDE_S`) overlaps an incremental GC with its own
#     GPU work, so the GC is effectively free — collect every step and keep memory minimal.
#     (Timing a GC in isolation would *overestimate* its cost here, because the async frees
#     pipeline with the next step; so we deliberately don't use `t_gc`.)
#   * A cheap step puts the GC on the critical path (the GPU is idle during it, so timing it in
#     isolation is accurate). Collect only ~every `t_gc/(ε·t_step)` steps to hold the amortized
#     cost near ε ≈ 2%.
# An unusually slow step (typically the backend's own memory reclaim under pressure) halves the
# interval to preempt further such stalls.
mutable struct AutoGCPacer <: GCPacer
    t_step::Float64   # EMA of step wall-time (seconds)
    t_gc::Float64     # EMA of one GC's wall-time (seconds)
    interval::Int     # current number of steps between collections
    since::Int        # steps since the last collection
end
AutoGCPacer() = AutoGCPacer(0.0, 0.0, 1, 0)

tic(::AutoGCPacer) = time_ns()

function maybe_gc!(p::AutoGCPacer, i::Integer, t0::UInt64)
    dt = (time_ns() - t0) / 1e9
    spike = p.t_step > 0.0 && dt > 3 * p.t_step
    p.t_step = p.t_step == 0.0 ? dt : 0.9 * p.t_step + 0.1 * dt
    base = if p.t_step >= GC_HIDE_S           # compute-bound: GC hides → every step
        1
    elseif p.t_gc == 0.0 || p.t_step == 0.0  # not measured yet
        1
    else                                     # cheap step: bound amortized GC cost to ε
        clamp(round(Int, p.t_gc / (GC_OVERHEAD * p.t_step)), 1, 4096)
    end
    p.interval = spike ? max(1, p.interval ÷ 2) : clamp(p.interval + 1, 1, base)
    p.since += 1
    if p.since >= p.interval
        g0 = time_ns()
        GC.gc(false)
        gdt = (time_ns() - g0) / 1e9
        p.t_gc = p.t_gc == 0.0 ? gdt : 0.9 * p.t_gc + 0.1 * gdt
        p.since = 0
    end
    return nothing
end

# Paced GC only helps with the caching allocator off (with it on, a step's buffers are pinned,
# so a mid-training GC reclaims nothing).
function _gc_pacer(gc_interval::Union{Integer, Symbol}, cache)
    cache === nothing || return NoGCPacer()
    gc_interval === :auto && return AutoGCPacer()
    gc_interval isa Integer && gc_interval > 0 && return FixedGCPacer(gc_interval)
    return NoGCPacer()   # gc_interval == 0: pacing disabled
end

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

    # Reactant / XLA fast path: compile the whole training step (forward + Enzyme reverse pass +
    # optimiser update) into one XLA executable, reused across batches of the same shape.
    if Flux.get_device_type(model) <: Flux.ReactantDevice
        model isa Duplicated && throw(ArgumentError(
            "`Duplicated` models are not supported on Reactant devices; pass the plain model \
             that already lives on the Reactant device."))
        (adtype isa AutoEnzyme || adtype isa AutoZygote) || @warn(
            "On a Reactant device `train!` always differentiates with Enzyme; ignoring \
             adtype=$adtype.", maxlog=1)
        return _reactant_train!(loss, model, data, opt)
    end

    Flux.trainmode!(model)
    cache = caching_allocator ? GPUArrays.AllocCache() : nothing
    pacer = _gc_pacer(gc_interval, cache)   # decides when to run an incremental GC (see above)

    @withprogress for (i,d) in enumerate(data)
        d_splat = d isa Tuple ? d : (d,)

        t0 = tic(pacer)
        # The first step is run without the cache on purpose. On a GPU it triggers cuDNN's
        # convolution-algorithm search, whose one-off probe workspaces are transient but
        # would be pinned by the cache — they are never reused and can be large enough to
        # blow past GPU memory.
        if cache === nothing || i == 1
            opt, model = _train_step!(loss, adtype, model, opt, d_splat, i)
        else
            # Reuse the memory allocated during the previous step, see issue #2523.
            GPUArrays.@cached cache begin
                opt, model = _train_step!(loss, adtype, model, opt, d_splat, i)
            end
        end

        maybe_gc!(pacer, i, t0)

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
