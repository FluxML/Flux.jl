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
    l, ∂L∂m = Flux.trainstep!(loss3, model, d, opt_state)
end
```
where each iteration is a single call to [`trainstep!`](@ref Flux.trainstep!).
You can also write this loop yourself, if you need more flexibility.
For this reason `train!` is not highly extensible.
It adds only a few features to the loop above:

* Stop with a `DomainError` if the loss is infinite or `NaN` at any point.

* Show a progress bar using [`@withprogress`](https://github.com/JuliaLogging/ProgressLogging.jl).

* Manage memory. Runs an incremental garbage collection adaptively.
"""
function train!(loss, adtype::AbstractADType, model, data, opt_state; cb = nothing,
                caching_allocator::Bool = false, gc_interval::Union{Integer, Symbol} = :auto)
    isnothing(cb) || error("""train! does not support callback functions.
                                For more control use a loop with `gradient` and `update!`.""")
    gc_interval isa Symbol && gc_interval !== :auto &&
        throw(ArgumentError("`gc_interval` must be a non-negative integer or `:auto`, got `:$gc_interval`"))

    Flux.trainmode!(model)

    # On a Reactant device the fused step is compiled by `trainstep!`; the caching allocator and the
    # paced GC below are CUDA-oriented (they manage `CuArray` reserved memory) and don't apply.
    on_reactant = _on_reactant(model)
    if on_reactant
        cache = nothing
        pacer = NoGCPacer()
    else
        cache = caching_allocator ? GPUArrays.AllocCache() : nothing
        pacer = _gc_pacer(gc_interval, cache)   # decides when to run an incremental GC (see above)
    end

    @withprogress for (i, batch) in enumerate(data)
        t0 = tic(pacer)
        # The first step is run without the cache on purpose. On a GPU it triggers cuDNN's
        # convolution-algorithm search, whose one-off probe workspaces are transient but
        # would be pinned by the cache — they are never reused and can be large enough to
        # blow past GPU memory.
        if cache === nothing || i == 1
            l = trainstep!(loss, adtype, model, batch, opt_state)
        else
            # Reuse the memory allocated during the previous step, see issue #2523.
            GPUArrays.@cached cache begin
                l = trainstep!(loss, adtype, model, batch, opt_state)
            end
        end

        # `trainstep!` mutates `model`/`opt_state` in place and skips the update itself on a
        # non-finite loss, so the model is intact here when we stop. (On a Reactant device the update
        # is fused into the executable and already applied; this only halts further steps.) `l` is the
        # loss value, or `(loss, aux...)` when the loss returns auxiliary outputs — guard on the scalar.
        ls = _loss_scalar(l)
        isfinite(ls) || throw(DomainError(lazy"Loss is $ls on data item $i, stopping training"))

        maybe_gc!(pacer, i, t0)

        @logprogress Base.haslength(data) ? i/length(data) : nothing
    end
    isnothing(cache) || GPUArrays.unsafe_free!(cache)
    return nothing
end

train!(loss, model, data, opt_state; kws...) =
    train!(loss, AutoZygote(), model, data, opt_state; kws...)

train!(loss, model::Duplicated, data, opt_state; kws...) =
    train!(loss, AutoEnzyme(), model, data, opt_state; kws...)

"""
    trainstep!(loss, [adtype,] model, batch, opt_state) -> loss

Perform a single optimisation step: differentiate the loss with respect to `model`,
update `model` and `opt_state` **in place** according to the rule encoded in `opt_state`, and return
the `loss` value. 

If `batch` is a `Tuple`, its elements are spliced into `loss` after the model, so the loss is
evaluated as `loss(model, batch...)`; any other `batch` is passed as-is, i.e. `loss(model, batch)`.

The optional `adtype` selects the automatic differentiation engine among
the ones supported by [`gradient`](@ref); if omitted, Zygote is used, unless `model` is a `Duplicated`
from Enzyme.jl, in which case Enzyme is used.

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
l, stats = Flux.trainstep!(loss, model, (x, y), opt_state)   # l == (loss, stats); stats.acc
```

## Non-finite loss

If the (scalar) loss comes out non-finite the update is skipped, leaving `model` unchanged (`train!`
turns this into a `DomainError`). On a Reactant device this guard does not apply, as the update is
fused into the compiled step.

## Reactant

When the `model` lives on a [Reactant](https://github.com/EnzymeAD/Reactant.jl) device, the whole step
(forward pass, Enzyme reverse pass and optimiser update) is compiled into a single XLA executable,
cached and reused across calls with the same model, optimiser, loss and batch shape. The returned
value — the loss, or `(loss, aux...)` — is read back to the host.

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
trainstep!(loss, model, batch, opt_state) =
    trainstep!(loss, AutoZygote(), model, batch, opt_state)

trainstep!(loss, model::Duplicated, batch, opt_state) =
    trainstep!(loss, AutoEnzyme(), model, batch, opt_state)

trainstep!(loss, adtype::AbstractADType, model, batch, opt_state) = 
    trainstep!(loss, adtype, model, (batch,), opt_state)

function trainstep!(loss, adtype::AbstractADType, model, batch::Tuple, opt_state)
    if _is_reactant_model(model, adtype)
        return _reactant_trainstep!(loss, model, batch, opt_state)
    end
    l, _ = _eager_step!(loss, adtype, model, batch, opt_state)
    return l
end

"""
    trainstep_withgradient!(loss, [adtype,] model, batch, opt_state) -> loss, grad

Like [`trainstep!`](@ref Flux.trainstep!), but also returns the `grad`ient of the loss with respect
to the model. Everything else is identical: `model` and `opt_state` are updated in place and the
first returned value is whatever the loss returned — the scalar loss, or `(loss, aux...)` when the
loss returns auxiliary outputs (see [`trainstep!`](@ref Flux.trainstep!)).

On a Reactant device the returned value (the loss, or `(loss, aux...)`) is read back to the host,
while the returned `grad` stays on the device. Returning the gradient makes it an output of the
compiled step, which raises peak memory relative to `trainstep!`; prefer `trainstep!` when the
gradient is not needed.
"""
trainstep_withgradient!(loss, model, batch, opt_state) =
    trainstep_withgradient!(loss, AutoZygote(), model, batch, opt_state)

trainstep_withgradient!(loss, model::Duplicated, batch, opt_state) =
    trainstep_withgradient!(loss, AutoEnzyme(), model, batch, opt_state)

trainstep_withgradient!(loss, adtype::AbstractADType, model, batch, opt_state) = 
    trainstep_withgradient!(loss, adtype, model, (batch,), opt_state)

function trainstep_withgradient!(loss, adtype::AbstractADType, model, batch::Tuple, opt_state)
    if _is_reactant_model(model, adtype)
        return _reactant_trainstep_withgradient!(loss, model, batch, opt_state)
    end
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
        p = _first_param(c)
        p === nothing || return p
    end
    return nothing
end

# Cheap "does this model live on a Reactant device" check (see `_first_param`).
function _on_reactant(model)
    p = _first_param(model)
    return p !== nothing && Flux.get_device_type(p) <: Flux.ReactantDevice
end

# On a Reactant device, validate the model/adtype and signal the caller to take the compiled path.
# The model's device is the canonical signal for Reactant training (it's what `train!` keys on too);
# the extension then checks the batch is device-resident, so keying on the model here keeps the
# "host-resident data" error meaningful.
function _is_reactant_model(model, adtype)
    _on_reactant(model) || return false
    model isa Duplicated && throw(ArgumentError(
        "`Duplicated` models are not supported on Reactant devices; pass the plain model \
         that already lives on the Reactant device."))
    (adtype isa AutoEnzyme || adtype isa AutoZygote) || @warn(
        "On a Reactant device training always differentiates with Enzyme; ignoring \
         adtype=$adtype.", maxlog=1)
    return true
end

_update!(opt_state, model, grads) = Optimisers.update!(opt_state, model, grads)

function _update!(opt_state, model::Duplicated, grad)
    opt_state, model2 = Optimisers.update!(opt_state, model.val, grad)
    return opt_state, Duplicated(model2, model.dval)
end


end # module Train
