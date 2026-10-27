module Train

using LinearAlgebra
using Optimisers: Optimisers
using Functors: fmap, fmapstructure, children
using ..Flux: Flux
using GPUArrays: GPUArrays

using ProgressLogging: @progress, @withprogress, @logprogress
using EnzymeCore: Duplicated
using ADTypes: AbstractADType, AutoEnzyme, AutoZygote

export setup, train!, trainstep!, trainstep_withgradient!

# Implemented by FluxReactantExt when Reactant.jl is loaded. Only reached from the ReactantDevice
# branch of `trainstep!`/`trainstep_withgradient!` below, which requires the model to hold Reactant
# arrays — i.e. Reactant (and thus the extension) is already loaded. The bare declarations let those
# functions reference them; a MethodError is the (essentially unreachable) safety net if the extension
# is missing.
function _reactant_trainstep! end
function _reactant_trainstep_withgradient! end

# ---------------------------------------------------------------------------
# Paced garbage collection for `train!` and `trainstep!`.
#
# With the caching allocator off, dead GPU buffers are only reclaimed by the GC, which seldom
# fires on its own here (the `CuArray` wrappers are tiny on the CPU heap), so reserved GPU
# memory creeps up (issue #2523). A periodic incremental GC bounds that growth at a fraction
# of the cost of collecting every step. A `GCPacer` decides, after each training step and
# given how long it took, whether to run `GC.gc(false)`. Each model has its own, in
# `STEP_PACERS`, used by both `trainstep!` and `train!` (see `_paced_step`).
# ---------------------------------------------------------------------------
abstract type GCPacer end

# No paced GC: with the caching allocator on a mid-step GC reclaims nothing (the step's
# buffers are pinned), and `gc_interval = 0` disables pacing explicitly.
struct NoGCPacer <: GCPacer end
maybe_gc!(::NoGCPacer, dt::Real, types::Tuple) = nothing

# Number of collections run by all pacers. A pacer counts its steps since the last of them,
# whichever pacer ran it, as that collection freed the garbage of the steps before it. So models
# trained in alternation (a GAN), or one per thread, don't each collect at their own cadence.
# Julia's automatic collections don't count: they run mid-step, while its arrays are in use.
const GC_COUNT = Threads.Atomic{Int}(0)

# The pacers have the fields `since` (steps since the last collection) and `seen` (`GC_COUNT`
# after their last step).
function count_step!(pacer::GCPacer)
    c = GC_COUNT[]
    if c != pacer.seen
        pacer.since = 0
        pacer.seen = c
    end
    pacer.since += 1
    return nothing
end

function run_gc!(pacer::GCPacer, full::Bool)
    GC.gc(full)
    pacer.seen = Threads.atomic_add!(GC_COUNT, 1) + 1
    pacer.since = 0
    return nothing
end

# The types of the arguments of the steps run so far (see `_step_types`). A step with new ones
# compiles, for seconds or minutes, and the arrays it allocates survive Julia's automatic
# collections during that time. That moves them to the old generation, which `GC.gc(false)`
# doesn't free (#2795), so such a step is followed by a full GC instead. A full GC costs ~0.1 s,
# too much to run after every step, or after the first step of every `train!` call when it is
# called once per batch. A lookup costs ~0.2 μs.
const STEP_TYPES = Set{Any}()
const STEP_TYPES_LOCK = ReentrantLock()

_step_types(loss, adtype, model, batch, opt_state) =
    (typeof(loss), typeof(adtype), typeof(model), typeof(batch), typeof(opt_state))

function _new_step_types!(types::Tuple)
    @lock STEP_TYPES_LOCK begin
        types in STEP_TYPES && return false
        push!(STEP_TYPES, types)
        return true
    end
end

# Fixed cadence: collect every `interval` steps.
mutable struct FixedGCPacer <: GCPacer
    interval::Int
    since::Int
    seen::Int
end
FixedGCPacer(interval::Integer) = FixedGCPacer(interval, 0, GC_COUNT[])

function maybe_gc!(pacer::FixedGCPacer, dt::Real, types::Tuple)
    count_step!(pacer)
    if _new_step_types!(types)
        run_gc!(pacer, true)
    elseif pacer.since >= pacer.interval
        run_gc!(pacer, false)
    end
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
    since::Int
    seen::Int
end
AutoGCPacer() = AutoGCPacer(0.0, 0.0, 1, 0, GC_COUNT[])

function maybe_gc!(pacer::AutoGCPacer, dt::Real, types::Tuple)
    count_step!(pacer)
    spike = pacer.t_step > 0.0 && dt > 3 * pacer.t_step
    pacer.t_step = pacer.t_step == 0.0 ? dt : 0.9 * pacer.t_step + 0.1 * dt
    base = if pacer.t_step >= GC_HIDE_S               # compute-bound: GC hides → every step
        1
    elseif pacer.t_gc == 0.0 || pacer.t_step == 0.0   # not measured yet
        1
    else                                              # cheap step: bound amortized GC cost to ε
        clamp(round(Int, pacer.t_gc / (GC_OVERHEAD * pacer.t_step)), 1, 4096)
    end
    pacer.interval = spike ? max(1, pacer.interval ÷ 2) : clamp(pacer.interval + 1, 1, base)
    if _new_step_types!(types)
        run_gc!(pacer, true)   # untimed: `t_gc` is the cost of an incremental GC
    elseif pacer.since >= pacer.interval
        g0 = time_ns()
        run_gc!(pacer, false)
        gdt = (time_ns() - g0) / 1e9
        pacer.t_gc = pacer.t_gc == 0.0 ? gdt : 0.9 * pacer.t_gc + 0.1 * gdt
    end
    return nothing
end

function _check_gc_interval(gc_interval::Union{Integer, Symbol})
    gc_interval isa Symbol && gc_interval !== :auto &&
        throw(ArgumentError("`gc_interval` must be a non-negative integer or `:auto`, got `:$gc_interval`"))
    return nothing
end

_pacing(gc_interval) = gc_interval === :auto || (gc_interval isa Integer && gc_interval > 0)

_gc_pacer(gc_interval) =
    !_pacing(gc_interval) ? NoGCPacer() :
    gc_interval === :auto ? AutoGCPacer() : FixedGCPacer(gc_interval)

_paces(pacer::AutoGCPacer, gc_interval) = gc_interval === :auto
_paces(pacer::FixedGCPacer, gc_interval) = gc_interval isa Integer && pacer.interval == gc_interval
_paces(::NoGCPacer, gc_interval) = false

# The pacers of `trainstep!`, keyed on the `objectid` of the model's first parameter array, with
# a `WeakRef` to that array. Keying on the loss or the batch wouldn't work: closures and batch
# shapes change between steps of the same loop. An entry whose array has been freed is stale, and
# is pruned when an entry is added (the same pattern as `COMPILE_CACHE` in FluxReactantExt).
const STEP_PACERS = Dict{UInt, Tuple{WeakRef, GCPacer}}()
const STEP_PACERS_LOCK = ReentrantLock()

_prune_step_pacers!() = filter!(kv -> kv.second[1].value !== nothing, STEP_PACERS)

# `ALLOC_CACHE` is a GPUArrays internal.
_in_alloc_cache() = isdefined(GPUArrays, :ALLOC_CACHE) && GPUArrays.ALLOC_CACHE[] !== nothing

# The pacer of a model whose first parameter is `param`, on a device of type `devtype`. Only an
# eager model gets one, on the CPU or a GPU: on Reactant XLA owns the buffers. On the CPU, the GC
# counts the arrays' memory and collects on its own, but mid-step, while the arrays are in use: it
# then runs full collections, which made compute-bound models 1.1–1.5× slower than with pacing.
# Inside `GPUArrays.@cached` the cache keeps the step's buffers, so a GC would free nothing. An
# immutable `param` (a `StaticArray`) can't identify its model.
function _step_pacer(param, devtype::Type, gc_interval::Union{Integer, Symbol})
    _check_gc_interval(gc_interval)
    (_pacing(gc_interval) && ismutable(param) && !_in_alloc_cache() &&
     devtype <: Union{Flux.CPUDevice, Flux.MLDataDevices.AbstractGPUDevice}) || return NoGCPacer()
    key = objectid(param)
    @lock STEP_PACERS_LOCK begin
        ref, pacer = get(STEP_PACERS, key, (WeakRef(), NoGCPacer()))
        ref.value === param && _paces(pacer, gc_interval) && return pacer
        _prune_step_pacers!()
        pacer = _gc_pacer(gc_interval)
        STEP_PACERS[key] = (WeakRef(param), pacer)
        return pacer
    end
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
    train!(loss, [adtype,] model, data, opt_state; gc_interval = :auto)

Uses a `loss` function and training `data` to improve the `model`'s parameters
according to a particular optimisation rule encoded in `opt_state`.

Iterates through `data` once, evaluating for each `d in data` either
`loss(model, d...)` if `d isa Tuple`, or else `loss(model, d)` for other `d`.

The optional argument `adtype` selects an automatic differentiation engine among the ones supported by
[`gradient`](@ref). If no `adtype` is given, Enzyme is used when `model` is a `Duplicated` from Enzyme.jl
or lives on a [Reactant](https://github.com/EnzymeAD/Reactant.jl) device, and Zygote is used otherwise.
On a Reactant device only Enzyme is supported, and any other `adtype` throws an `ArgumentError`
(see [`trainstep!`](@ref Flux.trainstep!)).

For example, with these definitions...
```
data = [(x1, y1), (x2, y2), (x3, y3)]

loss3(m, x, y) = norm(m(x) .- y)        # the model is the first argument

opt_state = Flux.setup(Adam(), model)   # explicit setup of optimiser momenta
```
...calling `Flux.train!(loss3, model, data, opt_state)` runs a loop much like this:
```
for d in data
    l = Flux.trainstep!(loss3, model, d, opt_state)
end
```
where each iteration is a single call to [`trainstep!`](@ref Flux.trainstep!).
You can also write this loop yourself, if you need more flexibility.
For this reason `train!` is not highly extensible.
It adds only a few features to the loop above:

* Stop with a `DomainError` if the loss is infinite or `NaN` at any point. The update for that
  batch is skipped, so the parameters and `opt_state` keep their values from before it, also on a
  Reactant device.

* Show a progress bar using [`@withprogress`](https://github.com/JuliaLogging/ProgressLogging.jl).

* Manage memory, by running garbage collections between steps (see below).

## Garbage collection

On a GPU, arrays are freed by the garbage collector, which seldom runs on its own, since they take
little CPU memory. So `train!` runs collections between steps, as [`trainstep!`](@ref Flux.trainstep!)
does, which keeps the GPU memory in use close to what one step needs. It runs them on the CPU too,
where they make larger models train faster. The keyword `gc_interval` sets how often:

* `:auto` (the default): an incremental collection after each step that takes more than a few
  milliseconds, and less often after faster steps, so that collections take about 2% of the
  training time;
* an integer `n`: an incremental collection every `n` steps;
* `0`: no collections.

Unless `gc_interval = 0`, a step that compiles, that is, the first one with the types of its
arguments (loss, model, batch and optimiser state), is followed by a full collection instead, as
incremental ones don't free the arrays of such a step. Collections are paced for each model, also
across calls of `train!` and `trainstep!` (see [`trainstep!`](@ref Flux.trainstep!)). There are none
on a Reactant device, nor inside `GPUArrays.@cached`, where they would free nothing; with
`caching_allocator = true`, the steps after the first one run there.
"""
function train!(loss, adtype::Union{Nothing, AbstractADType}, model, data, opt_state; cb = nothing,
                caching_allocator::Bool = false, gc_interval::Union{Integer, Symbol} = :auto)
    isnothing(cb) || error("""train! does not support callback functions.
                                For more control use a loop with `gradient` and `update!`.""")
    _check_gc_interval(gc_interval)

    Flux.trainmode!(model)

    # Resolve the AD backend and the Reactant flag exactly once here (not per batch), then thread both
    # into the per-step worker below. On a Reactant device the fused step is compiled; the caching
    # allocator is CUDA-oriented (it manages `CuArray` reserved memory) and doesn't apply.
    adtype, on_reactant, param, devtype = _resolve_backend(model, adtype)
    cache = caching_allocator && !on_reactant ? GPUArrays.AllocCache() : nothing

    @withprogress for (i, batch) in enumerate(data)
        # A `Tuple` batch is spliced into the loss (`loss(model, batch...)`), anything else is passed
        # as-is (`loss(model, batch)`) — the wrapping `trainstep!` would otherwise do. We call the
        # internal `_paced_step` directly so the resolved backend/Reactant flag are reused instead of
        # re-derived every batch.
        b = batch isa Tuple ? batch : (batch,)
        # The first step is run without the cache on purpose. On a GPU it triggers cuDNN's
        # convolution-algorithm search, whose one-off probe workspaces are transient but
        # would be pinned by the cache — they are never reused and can be large enough to
        # blow past GPU memory.
        if cache === nothing || i == 1
            l = _paced_step(_trainstep!, loss, adtype, on_reactant, param, devtype, model, b, opt_state,
                            gc_interval)
        else
            # Reuse the memory allocated during the previous step, see issue #2523. A GC would free
            # nothing here, so there is none.
            GPUArrays.@cached cache begin
                l = _paced_step(_trainstep!, loss, adtype, on_reactant, param, devtype, model, b,
                                opt_state, gc_interval)
            end
        end

        # `trainstep!` mutates `model`/`opt_state` in place and skips the update itself on a
        # non-finite loss (inside the compiled step on a Reactant device), so the model is intact here
        # when we stop. `l` is the loss value, or `(loss, aux...)` when the loss returns auxiliary
        # outputs — guard on the scalar.
        ls = _loss_scalar(l)
        isfinite(ls) || throw(DomainError(lazy"Loss is $ls on data item $i, stopping training"))

        @logprogress Base.haslength(data) ? i/length(data) : nothing
    end
    isnothing(cache) || GPUArrays.unsafe_free!(cache)
    return nothing
end

# No `adtype` given: pass the `nothing` sentinel so `_resolve_backend` picks the device-aware default
# (Enzyme for `Duplicated`/Reactant models, Zygote otherwise).
train!(loss, model, data, opt_state; kws...) =
    train!(loss, nothing, model, data, opt_state; kws...)

"""
    trainstep!(loss, [adtype,] model, batch, opt_state; gc_interval = :auto) -> loss

Perform a single optimisation step: differentiate the loss with respect to `model`,
update `model` and `opt_state` **in place** according to the rule encoded in `opt_state`, and return
the `loss` value. 

If `batch` is a `Tuple`, its elements are spliced into `loss` after the model, so the loss is
evaluated as `loss(model, batch...)`; any other `batch` is passed as-is, i.e. `loss(model, batch)`.

The optional `adtype` selects the automatic differentiation engine among the ones supported by
[`gradient`](@ref). If omitted, Enzyme is used when `model` is a `Duplicated` from Enzyme.jl or lives
on a Reactant device, and Zygote is used otherwise. On a Reactant device only Enzyme is supported,
and any other `adtype` throws an `ArgumentError`.

Use [`trainstep_withgradient!`](@ref Flux.trainstep_withgradient!) instead
if you also need the gradient.

[`train!`](@ref Flux.train!) is a loop over the data built on top of `trainstep!`.

## Auxiliary outputs

Like [`withgradient`](@ref Flux.withgradient), the loss may return auxiliary data alongside the
scalar loss: if it returns a `Tuple` or `NamedTuple` whose first element is the loss, the gradient
is taken of the loss alone and the whole value is returned. This is handy for logging metrics
computed during the forward pass (accuracy, a breakdown of loss terms, …):

```julia
function loss(m, x, y)
    ŷ = m(x)
    Flux.mse(ŷ, y), (; acc = mean(onecold(ŷ) .== onecold(y)))
end
l, stats = Flux.trainstep!(loss, model, (x, y), opt_state)   # l is the loss, stats.acc the accuracy
```

## Non-finite loss

If the (scalar) loss comes out non-finite the update is skipped, leaving `model` and `opt_state`
unchanged (`train!` turns this into a `DomainError`). The loss value, and with
[`trainstep_withgradient!`](@ref Flux.trainstep_withgradient!) the gradient, are still returned.
On a Reactant device the check runs inside the compiled step.

## Garbage collection

Like [`train!`](@ref Flux.train!), `trainstep!` runs garbage collections between steps, which keep
the GPU memory in use close to what one step needs, with the same keyword `gc_interval` (see
`train!` for its values). Since each call is a single step, they are paced for each model, across
calls. Each model has its own count of steps, which restarts after any collection, so models
trained in alternation, as in a GAN, don't each collect.

## Reactant

When the `model` lives on a [Reactant](https://github.com/EnzymeAD/Reactant.jl) device, the whole step
(forward pass, reverse pass and optimiser update) is compiled into a single XLA executable, cached and
reused across calls with the same model, optimiser, loss, backend and batch shape. The returned
value — the loss, or `(loss, aux...)` — is read back to the host. Differentiation uses Enzyme, the
only backend supported there: any other `adtype`, such as `AutoZygote()` or `AutoMooncake()`, throws
an `ArgumentError`.

# Example
```julia
model = Dense(2 => 1, tanh)
opt_state = Flux.setup(Adam(), model)

x, y = rand(Float32, 2, 8), rand(Float32, 1, 8)
loss(m, x, y) = Flux.mse(m(x), y)
trainmode!(model) # necessary for dropout/batchnorm layers
for epoch in 1:100
    l = Flux.trainstep!(loss, model, (x, y), opt_state)
    @info "epoch \$epoch" loss=l
end
```
"""
trainstep!(loss, model, batch, opt_state; kws...) =
    trainstep!(loss, nothing, model, batch, opt_state; kws...)

trainstep!(loss, adtype::Union{Nothing, AbstractADType}, model, batch, opt_state; kws...) =
    trainstep!(loss, adtype, model, (batch,), opt_state; kws...)

function trainstep!(loss, adtype::Union{Nothing, AbstractADType}, model, batch::Tuple, opt_state;
                    gc_interval::Union{Integer, Symbol} = :auto)
    adtype, on_reactant, param, devtype = _resolve_backend(model, adtype)
    return _paced_step(_trainstep!, loss, adtype, on_reactant, param, devtype, model, batch, opt_state,
                       gc_interval)
end

# Runs `step!` (`_trainstep!` or `_trainstep_withgradient!`), followed by the model's paced GC.
function _paced_step(step!, loss, adtype, on_reactant, param, devtype, model, batch, opt_state,
                     gc_interval)
    pacer = _step_pacer(param, devtype, gc_interval)
    t0 = time_ns()
    r = step!(loss, adtype, on_reactant, model, batch, opt_state)
    maybe_gc!(pacer, (time_ns() - t0) / 1e9, _step_types(loss, adtype, model, batch, opt_state))
    return r
end

# Single step with the backend already resolved (shared by `trainstep!` and the `train!` loop).
function _trainstep!(loss, adtype::AbstractADType, on_reactant::Bool, model, batch::Tuple, opt_state)
    on_reactant && return _reactant_trainstep!(loss, adtype, model, batch, opt_state)
    l, _ = _eager_step!(loss, adtype, model, batch, opt_state)
    return l
end

"""
    trainstep_withgradient!(loss, [adtype,] model, batch, opt_state; gc_interval = :auto) -> loss, grad

Like [`trainstep!`](@ref Flux.trainstep!), but also returns the `grad`ient of the loss with respect
to the model. Everything else is identical: `model` and `opt_state` are updated in place and the
first returned value is whatever the loss returned — the scalar loss, or `(loss, aux...)` when the
loss returns auxiliary outputs (see [`trainstep!`](@ref Flux.trainstep!)).

On a Reactant device the returned value (the loss, or `(loss, aux...)`) is read back to the host,
while the returned `grad` stays on the device. Returning the gradient makes it an output of the
compiled step, which raises peak memory relative to `trainstep!`; prefer `trainstep!` when the
gradient is not needed.
"""
trainstep_withgradient!(loss, model, batch, opt_state; kws...) =
    trainstep_withgradient!(loss, nothing, model, batch, opt_state; kws...)

trainstep_withgradient!(loss, adtype::Union{Nothing, AbstractADType}, model, batch, opt_state; kws...) =
    trainstep_withgradient!(loss, adtype, model, (batch,), opt_state; kws...)

function trainstep_withgradient!(loss, adtype::Union{Nothing, AbstractADType}, model, batch::Tuple, opt_state;
                                 gc_interval::Union{Integer, Symbol} = :auto)
    adtype, on_reactant, param, devtype = _resolve_backend(model, adtype)
    return _paced_step(_trainstep_withgradient!, loss, adtype, on_reactant, param, devtype, model,
                       batch, opt_state, gc_interval)
end

# Single step with the backend already resolved; like `_trainstep!` but returns `(val, grad)`.
function _trainstep_withgradient!(loss, adtype::AbstractADType, on_reactant::Bool, model, batch::Tuple, opt_state)
    on_reactant && return _reactant_trainstep_withgradient!(loss, adtype, model, batch, opt_state)
    return _eager_step!(loss, adtype, model, batch, opt_state)
end

# Eager (Zygote/Enzyme) single step, shared by both entry points; returns `(val, grad)` where `val`
# is whatever the loss returned (a scalar, or a Tuple/NamedTuple `(loss, aux...)` — see
# `_loss_scalar`). The update is skipped on a non-finite loss so the model is left uncorrupted, and
# the returned value lets the caller (e.g. `train!`) decide how to react.
function _eager_step!(loss, adtype, model, batch::Tuple, opt_state)
    v, gs = Flux.withgradient(m -> loss(m, batch...), adtype, model)
    isfinite(_loss_scalar(v)) && _update!(opt_state, model, gs[1])
    return v, gs[1]
end

# The scalar loss out of a `trainstep!`/`withgradient` value. When the loss returns auxiliary
# outputs (a Tuple or NamedTuple whose first element is the loss) the whole thing is returned to the
# user, but the finiteness guard and `train!` only look at the scalar.
_loss_scalar(v::Union{Tuple, NamedTuple}) = first(v)
_loss_scalar(v) = v

# First parameter array of a model — a cheap proxy for the model's device, since a model's parameters
# share a device in practice. `Flux.get_device_type(model)` traverses *every* parameter (microseconds
# for a large model, and it runs on every `trainstep!`), whereas one leaf suffices. Returns `nothing`
# for a model with no array parameters.
_first_param(x::AbstractArray{<:Number}) = x
_first_param(x::Duplicated) = _first_param(x.val)
function _first_param(x)
    for c in children(x)
        param = _first_param(c)
        param === nothing || return param
    end
    return nothing
end

# Resolve the AD backend and the Reactant flag in a single pass, finding the model's device exactly
# once per top-level call (see `_first_param`). Also returns the model's first parameter and the type
# of its device (`Nothing` for a model with no array parameters), for `trainstep!`'s GC pacer. When
# `adtype === nothing` (the caller passed no backend) the default is Enzyme
# for a `Duplicated` model or a Reactant-resident model, and Zygote otherwise; an explicit `adtype` is
# returned unchanged. The compiled Reactant step differentiates with Enzyme only, so any other backend
# is rejected here, before anything is traced. The model's device is the canonical Reactant signal
# (it's what `train!` keys on too); the extension then checks the batch is device-resident, so keying
# on the model here keeps the "host-resident data" error meaningful.
function _resolve_backend(model, adtype::Union{Nothing, AbstractADType})
    param = _first_param(model)
    devtype = param === nothing ? Nothing : Flux.get_device_type(param)
    on_reactant = devtype <: Flux.ReactantDevice
    model isa Duplicated && on_reactant && throw(ArgumentError(
        "`Duplicated` models are not supported on Reactant devices; pass the plain model \
         that already lives on the Reactant device."))
    on_reactant && !(adtype === nothing || adtype isa AutoEnzyme) && throw(ArgumentError(
        "Training a model on a Reactant device supports only the Enzyme AD backend, got `$adtype`. \
         Pass `AutoEnzyme()` or omit the `adtype` argument."))
    resolved = adtype === nothing ?
        ((on_reactant || model isa Duplicated) ? AutoEnzyme() : AutoZygote()) :
        adtype
    return resolved, on_reactant, param, devtype
end

_update!(opt_state, model, grads) = Optimisers.update!(opt_state, model, grads)

function _update!(opt_state, model::Duplicated, grad)
    opt_state, model2 = Optimisers.update!(opt_state, model.val, grad)
    return opt_state, Duplicated(model2, model.dval)
end


end # module Train
