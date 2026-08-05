module FluxReactantExt

using Flux: Flux, Train
using Optimisers: Optimisers
using ADTypes: AutoEnzyme
using Functors: fmapstructure
import Reactant

# Fixed-arity compiled steps: the whole batch is passed as ONE tuple (splat inside), so the thunk
# arity is always 4 and stable regardless of how many arrays a batch holds. They mirror Flux's own
# eager step (src/train.jl); the loss is a free by-product (`withgradient` already runs a
# reverse-with-primal pass) and both mutate `opt`/`model` in place inside the executable.
#
# `train_step!` uses the loss-only variant; `train_step_withgradient!` uses the one that also returns
# the gradient. They are kept separate on purpose: returning `grad` makes it an *output* of the
# executable, so XLA must keep the gradient buffers live to the end of the step instead of freeing
# each one right after the update reads it. That blocks buffer reuse and raises peak GPU memory
# (~+0.8 GiB / +30–60% for a ResNet-18). Keeping a loss-only variant lets `train!` (and anyone who
# doesn't need the gradient) avoid that cost.
function _reactant_step!(loss, model, data::Tuple, opt)
    l, gs = Flux.withgradient(m -> loss(m, data...), AutoEnzyme(), model)
    Optimisers.update!(opt, model, gs[1])
    return l
end

function _reactant_step_withgradient!(loss, model, data::Tuple, opt)
    l, gs = Flux.withgradient(m -> loss(m, data...), AutoEnzyme(), model)
    Optimisers.update!(opt, model, gs[1])
    return l, gs[1]
end

# Cache compiled steps so repeated calls (epochs, warm-up + timed runs) reuse the executable instead
# of recompiling (Reactant has no global compile cache). Model/optimiser are runtime inputs, so the
# key only needs what the executable is specialised to: the concrete model (`objectid` + `typeof`,
# since parameter *shapes* aren't in the type), the optimiser rule type, the loss identity, and the
# batch element types + sizes.
#
# Auto-eviction: each entry keeps a `WeakRef` to one of the model's parameter arrays and dead entries
# are pruned on lookup, so an executable is freed once its model is garbage-collected (verified: the
# executable holds no strong reference to the model's arrays). We can't key a `WeakKeyDict` on the
# model — it hashes by parameter *contents*, which change every in-place update — nor weak-reference
# the immutable model directly, hence the parameter array as the lifetime handle.
const COMPILE_CACHE = Dict{Any, Tuple{WeakRef, Any}}()

# One of the model's parameter arrays, used as the weak handle whose lifetime tracks the model's;
# `nothing` for a model with no array parameters (then that entry simply never auto-evicts).
function _evict_handle(model)
    for x in Optimisers.trainables(model)
        return x
    end
    return nothing
end

# Drop entries whose model has been garbage-collected, freeing their compiled executables.
function _prune_compile_cache!()
    isempty(COMPILE_CACHE) && return nothing
    dead = Any[]
    for (k, (ref, _)) in COMPILE_CACHE
        ref.value === nothing && push!(dead, k)
    end
    for k in dead
        delete!(COMPILE_CACHE, k)
    end
    return nothing
end

# A hashable summary of a batch element's array-leaf shapes (a differently-shaped batch needs its own
# executable). Recursing through the functor leaves handles any batch the loss accepts — not just
# arrays, but named tuples, `@functor` structs such as graphs, etc. Non-array leaves are kept as-is,
# matching the constants Reactant bakes in. The array method is a fast path for the common case.
_shape_signature(x::AbstractArray) = size(x)
_shape_signature(x) = fmapstructure(a -> a isa AbstractArray ? size(a) : a, x)

# `withgradient` selects the loss-only executable or the one that also returns the gradient, and is
# part of the key so the two variants for the same model are cached separately.
function _compiled_step(loss, model, dtuple, opt, withgradient::Bool)
    _prune_compile_cache!()
    key = (withgradient, objectid(model), typeof(model), typeof(opt), objectid(loss),
           map(typeof, dtuple), map(_shape_signature, dtuple))
    entry = get(COMPILE_CACHE, key, nothing)
    entry === nothing || return entry[2]
    exe = withgradient ?
        Reactant.@compile(_reactant_step_withgradient!(loss, model, dtuple, opt)) :
        Reactant.@compile(_reactant_step!(loss, model, dtuple, opt))
    handle = _evict_handle(model)
    # A param-less model has nothing to weak-reference; fall back to the executable itself (which the
    # entry keeps alive) so such an entry is simply never pruned.
    COMPILE_CACHE[key] = (WeakRef(handle === nothing ? exe : handle), exe)
    return exe
end

_require_device_batch(batch) =
    Flux.get_device_type(batch) <: Flux.ReactantDevice || throw(ArgumentError(
        "`train_step!`/`train!` on a Reactant model requires device-resident data; move each batch to \
         the model's device first, e.g. `data = [(x, y) |> reactant_device() for (x, y) in data]`."))

# Reactant/XLA implementations of a single step, dispatched to from the ReactantDevice branch of
# `train_step!` / `train_step_withgradient!` (src/train.jl); `Flux.train!` loops over the loss-only
# one. The model and optimiser state already live on the Reactant device, and the batch is required to
# be device-resident too. The fused step is compiled once per distinct (model, optimiser, loss,
# batch-shape) and reused, mutating `opt_state`/`model` in place. The host loss read forces the async
# device→host sync; any returned gradient stays on the device.
function Train._reactant_train_step!(loss, model, batch::Tuple, opt_state)
    _require_device_batch(batch)
    step = _compiled_step(loss, model, batch, opt_state, false)
    l = step(loss, model, batch, opt_state)
    return Reactant.to_number(l)
end

function Train._reactant_train_step_withgradient!(loss, model, batch::Tuple, opt_state)
    _require_device_batch(batch)
    step = _compiled_step(loss, model, batch, opt_state, true)
    l, g = step(loss, model, batch, opt_state)
    return Reactant.to_number(l), g
end

end # module
