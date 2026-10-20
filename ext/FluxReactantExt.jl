module FluxReactantExt

using Flux: Flux, Train, cpu
using Optimisers: Optimisers
using Functors: Functors, fmapstructure, isleaf
using ADTypes: AutoEnzyme
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

# Compute the full loss value (loss + any aux) and the gradient of the scalar loss w.r.t. the model,
# with Enzyme (other backends are rejected by `Train._resolve_backend` before tracing).
#
# Enzyme's reverse mode returns only the scalar primal, so Flux's eager `withgradient` smuggles aux
# out through a mutable wrapper as a side effect — which does NOT survive Reactant's tracing (the aux
# would be silently dropped). Instead we differentiate `_reactant_objective` under Enzyme's Reactant
# ABI, which returns the aux as a real primal output of the one differentiated forward.
function _reactant_valgrad(loss, ::AutoEnzyme, model, data::Tuple)
    # `Val(true)`: copy inactive (integer) arrays into the shadow rather than sharing them. Reactant's
    # `autodiff` writes a (zero) gradient into the shadow of each such array, which would otherwise
    # reset a Dropout RNG seed after every step and freeze the mask.
    dmodel = Enzyme.make_zero(model, Val(true))
    ad = Enzyme.set_abi(Enzyme.ReverseWithPrimal, Reactant.ReactantABI)
    _, (_, v) = Enzyme.autodiff(ad, Enzyme.Const(_reactant_objective),
                                Enzyme.Duplicated, Enzyme.Const(loss),
                                Enzyme.Duplicated(model, dmodel), Enzyme.Const(data))
    return v, _reactant_grad(dmodel)
end

# Like the eager step, skip the update on a non-finite loss so `model` and `opt_state` stay intact.
# A traced conditional keeps the update in place; selecting between old and new values instead makes
# XLA copy the optimiser state every step. `track_numbers=false` keeps plain fields such as
# `Leaf.frozen::Bool` untraced.
function _reactant_update!(opt_state, model, v, g)
    l = _reactant_loss_scalar(v)
    Reactant.@trace track_numbers=false if isfinite(l)
        Optimisers.update!(opt_state, model, g)
    end
    return nothing
end

function _reactant_step!(loss, adtype, model, data::Tuple, opt_state)
    v, g = _reactant_valgrad(loss, adtype, model, data)
    _reactant_update!(opt_state, model, v, g)
    return v
end

function _reactant_step_withgradient!(loss, adtype, model, data::Tuple, opt_state)
    v, g = _reactant_valgrad(loss, adtype, model, data)
    _reactant_update!(opt_state, model, v, g)
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
# Guards `COMPILE_CACHE` against concurrent `trainstep!` calls from several threads. Held while
# compiling, so two tasks never compile the same step twice.
const COMPILE_CACHE_LOCK = ReentrantLock()

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
# Device numbers are runtime inputs like arrays, so only their type matters (they hash by identity).
_shape_signature(x::AbstractArray) = size(x)
_shape_signature(x::Reactant.RNumber) = typeof(x)
_shape_signature(x) = fmapstructure(_shape_leaf, x; exclude = a -> isleaf(a) || a isa Reactant.RNumber)
_shape_leaf(a) = a isa AbstractArray ? size(a) : a isa Reactant.RNumber ? typeof(a) : a

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
_collect_arrays!(acc, x::_DeviceLeaf) = push!(acc, x)
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
# in the key too, as `adtype` is an argument of the executable.
function _compiled_step(loss, adtype, model, dtuple, opt_state, withgradient::Bool)
    key = (withgradient, typeof(adtype), objectid(model), typeof(model), typeof(opt_state),
           objectid(loss), _loss_world(loss), map(typeof, dtuple), map(_shape_signature, dtuple),
           _alias_signature(dtuple), _static_signature(model), _frozen_signature(opt_state))
    @lock COMPILE_CACHE_LOCK begin
        _prune_compile_cache!()
        entry = get(COMPILE_CACHE, key, nothing)
        entry === nothing || return entry[2]
        _check_opt_state(opt_state)
        exe = withgradient ?
            Reactant.@compile(_reactant_step_withgradient!(loss, adtype, model, dtuple, opt_state)) :
            Reactant.@compile(_reactant_step!(loss, adtype, model, dtuple, opt_state))
        handle = _evict_handle(model)
        # A param-less model has nothing to weak-reference; fall back to the executable itself (which
        # the entry keeps alive) so such an entry is simply never pruned.
        COMPILE_CACHE[key] = (WeakRef(handle === nothing ? exe : handle), exe)
        length(COMPILE_CACHE) > COMPILE_CACHE_WARN && @warn(
            "Flux's Reactant training-step cache now holds $(length(COMPILE_CACHE)) compiled \
             executables. Each distinct (model, optimiser, loss, batch-shape) combination compiles and \
             caches its own step; steady growth usually means one of these changes every iteration \
             (e.g. a freshly built model or optimiser). Entries are freed automatically once their \
             model is garbage-collected.")
        return exe
    end
end

# `setup` on a Reactant model wraps each rule in Optimisers' `ReactantOptimiser`, which keeps the
# hyperparameters and scalar state (e.g. Adam's βt) on the device. A state set up on the host instead
# fails to compile, or bakes its scalars into the executable, so that `adjust!` is ignored and
# RAdam's step counter never advances.
# Checked only when compiling, as the rule type is part of the cache key.
function _check_opt_state(opt_state)
    ext = Base.get_extension(Optimisers, :OptimisersReactantMLDataDevicesExt)
    (ext === nothing || !isdefined(ext, :ReactantOptimiser)) && return nothing
    for ℓ in Functors.fleaves(opt_state; exclude = x -> x isa Optimisers.Leaf)
        ℓ isa Optimisers.Leaf && !(ℓ.rule isa ext.ReactantOptimiser) && throw(ArgumentError(
            "The optimiser state was not set up for this Reactant model (found rule $(ℓ.rule)). \
             Call `opt_state = Flux.setup(rule, model)` after moving the model to the device; to \
             resume from a saved optimiser state, then load it with \
             `Flux.loadmodel!(opt_state, saved_opt_state)`."))
    end
    return nothing
end

# `loadmodel!` into an optimiser state rebuilds its scalars (e.g. Adam's βt): keep them on `dst`'s
# client and device, as the default `oftype` would put them on Reactant's default device.
Flux._loadscalar(dst::Reactant.ConcretePJRTNumber{T}, src::Number) where {T} =
    Reactant.ConcretePJRTNumber{T}(src; client = Reactant.XLA.client(dst),
                                   device = Reactant.XLA.device(dst), sharding = dst.sharding)

# A traced integer array has eltype `TracedRNumber{<:Integer}`, so it needs its own method to be
# recognised as class labels. Its values are unknown at trace time and can't be checked: an
# out-of-range label silently picks a neighbouring entry.
Flux.Losses._islabels(ŷ::AbstractArray, y::Reactant.AnyTracedRArray{<:Integer}, dims::Integer) =
    ndims(y) == ndims(ŷ) - 1
Flux.Losses._check_label_values(::Reactant.AnyTracedRArray, C) = nothing
# Reactant's `NNlib.gather` rejects the lazy `vec` of a traced array; indexing traces to the same
# `stablehlo.gather`.
Flux.Losses._gather(x::Reactant.AnyTracedRArray, idx::AbstractArray) = x[idx]

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
