module FluxReactantExt

using Flux: Flux, Train
using Optimisers: Optimisers
using ADTypes: AutoEnzyme
using Functors: fmapstructure
import Reactant

# Fixed-arity compiled step: the whole batch is passed as ONE tuple (splat inside), so the
# thunk arity is always 4 and stable regardless of how many arrays a batch holds. Mirrors
# Flux's own eager `train_step!` (src/train.jl), returning `(loss, grad)`: the primal loss as an
# on-device scalar (free — `withgradient` already runs a reverse-with-primal pass) and the gradient
# w.r.t. the model (already computed for the update). `_reactant_train_step!` reads the loss back to
# the host for the `isfinite` guard and returns the gradient to the caller.
#
# Memory note: returning `grad` makes it an *output* of the executable, so XLA must keep the gradient
# buffers live to the end of the step instead of freeing each one right after the update reads it.
# That blocks buffer reuse and raises peak GPU memory (~+0.8 GiB / +30–60% for a ResNet-18 vs. a
# loss-only step). This is a deliberate tradeoff so `train!` and the public `train_step!` share one
# compiled step; if `train!`'s Reactant memory ever needs to drop back, give it a loss-only variant.
function _reactant_step!(loss, model, data::Tuple, opt)
    l, gs = Flux.withgradient(m -> loss(m, data...), AutoEnzyme(), model)
    Optimisers.update!(opt, model, gs[1])   # folded into the executable; mutates in place
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

function _compiled_step(loss, model, dtuple, opt)
    _prune_compile_cache!()
    key = (objectid(model), typeof(model), typeof(opt), objectid(loss),
           map(typeof, dtuple), map(_shape_signature, dtuple))
    entry = get(COMPILE_CACHE, key, nothing)
    entry === nothing || return entry[2]
    exe = Reactant.@compile _reactant_step!(loss, model, dtuple, opt)
    handle = _evict_handle(model)
    # A param-less model has nothing to weak-reference; fall back to the executable itself (which the
    # entry keeps alive) so such an entry is simply never pruned.
    COMPILE_CACHE[key] = (WeakRef(handle === nothing ? exe : handle), exe)
    return exe
end

# Reactant/XLA implementation of a single `Flux.train_step!`, dispatched to from the ReactantDevice
# branch of the eager `train_step!` (src/train.jl); `Flux.train!` loops over it. The model and
# optimiser state already live on the Reactant device, and the batch is required to be device-resident
# too. The fused step is compiled once per distinct (model, optimiser, loss, batch-shape) and reused,
# mutating `opt_state`/`model` in place. Returns the host-scalar loss (read forces the async
# device→host sync) and the on-device gradient.
function Train._reactant_train_step!(loss, model, batch::Tuple, opt_state)
    Flux.get_device_type(batch) <: Flux.ReactantDevice || throw(ArgumentError(
        "`train_step!`/`train!` on a Reactant model requires device-resident data; move each batch to \
         the model's device first, e.g. `data = [(x, y) |> reactant_device() for (x, y) in data]`."))
    step = _compiled_step(loss, model, batch, opt_state)
    l, g = step(loss, model, batch, opt_state)   # opt_state/model updated in place inside the executable
    return Reactant.to_number(l), g
end

end # module
