module FluxReactantExt

using Flux: Flux, Train, cpu
using Optimisers: Optimisers
using Functors: Functors, fmapstructure, isleaf
using ADTypes: AbstractADType, AutoEnzyme
import Reactant
using Reactant: Enzyme

# The scalar loss out of a loss value: the whole thing for a scalar loss, or `first` when the loss
# returns auxiliary outputs (a Tuple/NamedTuple `(loss, aux...)`). Checked at trace time, since
# Enzyme would otherwise silently differentiate the sum of an array-valued loss.
_reactant_loss_scalar(v::Union{Tuple, NamedTuple}) = _reactant_loss_scalar(first(v))
_reactant_loss_scalar(v::Number) = v
_reactant_loss_scalar(v) = throw(ArgumentError(
    "the loss must return a scalar, or a Tuple/NamedTuple whose first element is the scalar \
     loss; got $(typeof(v))"))

# Objective differentiated by `_reactant_valgrad`. The gradient is taken of the scalar loss only, so
# any auxiliary outputs are wrapped in `Reactant.ignore_derivatives` and handed back as a second,
# non-differentiated return value — a genuine primal output of the single `ReverseWithPrimal` pass.
function _reactant_objective(loss, model, data)
    v = loss(model, data...)
    return _reactant_loss_scalar(v), Reactant.ignore_derivatives(v)
end

# Zygote-style stripping of an Enzyme shadow model to a gradient: numeric leaves as-is, everything
# else (non-trainable state, activation functions, …) to `nothing`. Matches `Flux.withgradient`'s
# Enzyme path so the returned gradient and `Optimisers.update!` behave identically.
_reactant_grad(dmodel) =
    fmapstructure(x -> Optimisers.isnumeric(x) ? x : nothing, dmodel; prune=nothing)

# Compute the full loss value (loss + any aux) and the gradient of the scalar loss w.r.t. the model.
# The AD backend is selected by `adtype`.
#
# Default / `AutoEnzyme`: Enzyme's reverse mode returns only the scalar primal, so Flux's eager
# `withgradient` smuggles aux out through a mutable wrapper as a side effect — which does NOT survive
# Reactant's tracing (the aux would be silently dropped). Instead we differentiate `_reactant_objective`
# under Enzyme's Reactant ABI, which returns the aux as a real primal output of the one differentiated
# forward.
function _reactant_valgrad(loss, ::AutoEnzyme, model, data::Tuple)
    dmodel = Enzyme.make_zero(model)
    ad = Enzyme.set_abi(Enzyme.ReverseWithPrimal, Reactant.ReactantABI)
    _, (_, v) = Enzyme.autodiff(ad, Enzyme.Const(_reactant_objective),
                                Enzyme.Duplicated, Enzyme.Const(loss),
                                Enzyme.Duplicated(model, dmodel), Enzyme.Const(data))
    return v, _reactant_grad(dmodel)
end

# Any other backend (Zygote, Mooncake, …): trace `Flux.withgradient` through the compiled step. Unlike
# Enzyme's low-level `autodiff` above, `withgradient` for these backends already returns the full loss
# value (`res.val`, including any aux) as a genuine output, so it survives tracing without the
# `_reactant_objective`/`ignore_derivatives` machinery. Whether XLA can actually trace the backend is
# up to Reactant's op coverage — this makes the attempt reachable per the honour-explicit-adtype rule.
function _reactant_valgrad(loss, adtype::AbstractADType, model, data::Tuple)
    res = Flux.withgradient(m -> loss(m, data...), adtype, model)
    return res.val, res.grad[1]
end

function _reactant_step!(loss, adtype, model, data::Tuple, opt_state)
    v, g = _reactant_valgrad(loss, adtype, model, data)
    Optimisers.update!(opt_state, model, g)
    return v
end

function _reactant_step_withgradient!(loss, adtype, model, data::Tuple, opt_state)
    v, g = _reactant_valgrad(loss, adtype, model, data)
    Optimisers.update!(opt_state, model, g)
    return v, g
end

# Cache compiled steps so repeated calls (epochs, warm-up + timed runs) reuse the executable instead
# of recompiling (Reactant has no global compile cache). Model/optimiser are runtime inputs, so the
# key only needs what the executable is specialised to: the concrete model (`objectid` + `typeof`,
# since parameter *shapes* aren't in the type), the optimiser rule type, the loss identity and
# definition, the batch element types + sizes + aliasing, and the values traced as constants (layer
# modes, frozen parameters).
#
# Auto-eviction: each entry keeps a `WeakRef` to one of the model's parameter arrays and dead entries
# are pruned on lookup, so an executable is freed once its model is garbage-collected (verified: the
# executable holds no strong reference to the model's arrays). We can't key a `WeakKeyDict` on the
# model — it hashes by parameter *contents*, which change every in-place update — nor weak-reference
# the immutable model directly, hence the parameter array as the lifetime handle.
const COMPILE_CACHE = Dict{Any, Tuple{WeakRef, Any}}()

# Warn once the number of live cached executables passes this, on each further addition: a healthy
# training loop reuses a handful of steps, so unbounded growth usually means a new model/optimiser
# (or batch shape) is compiled every iteration by mistake.
const COMPILE_CACHE_WARN = 10

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

# Reactant passes device arrays and numbers as runtime inputs, but bakes every other value it meets
# while tracing into the executable. For the model these are e.g. the `active` flag that
# `trainmode!`/`testmode!` set, or a `Dropout` probability, so the key records their values.
const _DeviceLeaf = Union{AbstractArray{<:Number}, Reactant.RNumber}
_baked(x::_DeviceLeaf) = typeof(x)
_baked(x) = x
_static_signature(model) = fmapstructure(_baked, model; exclude = x -> isleaf(x) || x isa _DeviceLeaf)

# `freeze!`/`thaw!` flip the `frozen` flag of the optimiser state's leaves, which is baked in too.
_frozen_signature(x::Optimisers.Leaf, h::UInt = zero(UInt)) = hash(x.frozen, h)
_frozen_signature(x::Union{Tuple, NamedTuple, AbstractArray}, h::UInt = zero(UInt)) =
    foldl((h, y) -> _frozen_signature(y, h), x; init = h)
_frozen_signature(x, h::UInt = zero(UInt)) = h

# Reactant traces an array passed twice as a single input, so a step compiled for `(x, x)` must not
# be reused for `(x, y)`: record which batch arrays are the same object.
_collect_arrays!(acc, x::AbstractArray{<:Number}) = push!(acc, x)
_collect_arrays!(acc, x) = (isleaf(x) || foreach(c -> _collect_arrays!(acc, c), Functors.children(x)); acc)
function _alias_signature(dtuple::Tuple)
    arrays = foldl(_collect_arrays!, dtuple; init = Any[])
    return [findfirst(b -> b === a, arrays) for a in arrays]
end

# The world in which the loss' newest method was defined, so that redefining it recompiles. Methods
# the loss calls are not tracked.
_loss_world(loss) = maximum(m -> m.primary_world, methods(loss); init = zero(UInt))

# `withgradient` selects the loss-only executable or the one that also returns the gradient, and is
# part of the key so the two variants for the same model are cached separately. `typeof(adtype)` is
# in the key too: the compiled HLO is specialised to the differentiation path (Enzyme+ReactantABI vs
# a traced Zygote/Mooncake pass), so mixing backends on the same model/shapes must not alias.
function _compiled_step(loss, adtype, model, dtuple, opt_state, withgradient::Bool)
    _prune_compile_cache!()
    key = (withgradient, typeof(adtype), objectid(model), typeof(model), typeof(opt_state),
           objectid(loss), _loss_world(loss), map(typeof, dtuple), map(_shape_signature, dtuple),
           _alias_signature(dtuple), _static_signature(model), _frozen_signature(opt_state))
    entry = get(COMPILE_CACHE, key, nothing)
    entry === nothing || return entry[2]
    exe = withgradient ?
        Reactant.@compile(_reactant_step_withgradient!(loss, adtype, model, dtuple, opt_state)) :
        Reactant.@compile(_reactant_step!(loss, adtype, model, dtuple, opt_state))
    handle = _evict_handle(model)
    # A param-less model has nothing to weak-reference; fall back to the executable itself (which the
    # entry keeps alive) so such an entry is simply never pruned.
    COMPILE_CACHE[key] = (WeakRef(handle === nothing ? exe : handle), exe)
    length(COMPILE_CACHE) > COMPILE_CACHE_WARN && @warn(
        "Flux's Reactant training-step cache now holds $(length(COMPILE_CACHE)) compiled executables. \
         Each distinct (model, optimiser, loss, batch-shape) combination compiles and caches its own \
         step; steady growth usually means one of these changes every iteration (e.g. a freshly built \
         model or optimiser). Entries are freed automatically once their model is garbage-collected.")
    return exe
end

_require_device_batch(batch) =
    Flux.get_device_type(batch) <: Flux.ReactantDevice || throw(ArgumentError(
        "`trainstep!`/`train!` on a Reactant model requires device-resident data; move each batch to \
         the model's device first, e.g. `data = [(x, y) |> reactant_device() for (x, y) in data]`."))

# Reactant/XLA implementations of a single step, dispatched to from the ReactantDevice branch of
# `trainstep!` / `trainstep_withgradient!` (src/train.jl);
function Train._reactant_trainstep!(loss, adtype, model, batch::Tuple, opt_state)
    _require_device_batch(batch)
    step = _compiled_step(loss, adtype, model, batch, opt_state, false)
    l = step(loss, adtype, model, batch, opt_state)
    return cpu(l)
end

function Train._reactant_trainstep_withgradient!(loss, adtype, model, batch::Tuple, opt_state)
    _require_device_batch(batch)
    step = _compiled_step(loss, adtype, model, batch, opt_state, true)
    l, g = step(loss, adtype, model, batch, opt_state)
    return cpu(l), g
end

end # module
