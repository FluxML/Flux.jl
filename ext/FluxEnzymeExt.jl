module FluxEnzymeExt

using Flux

import Optimisers
import Functors
import Enzyme
using Enzyme: EnzymeCore, EnzymeRules, Active, Const, Duplicated, autodiff, ReverseWithPrimal, Reverse

EnzymeRules.inactive(::typeof(Flux.Losses._check_sizes), args...) = true

### gradient & withgradient
function Flux.gradient(f::F, adtype::AutoEnzyme, x::Vararg{Any,N}; zero::Bool=true) where {F,N}
    return _enzyme_gradient(f, map(_trymake_duplicated, x)...; zero)
end

function Flux.withgradient(f::F, adtype::AutoEnzyme, x::Vararg{Any,N}; zero::Bool=true) where {F,N}
    return _enzyme_withgradient(f, map(_trymake_duplicated, x)...; zero)
end

_trymake_duplicated(x::EnzymeCore.Duplicated) = x
_trymake_duplicated(x::EnzymeCore.Const) = x
_trymake_duplicated(x::EnzymeCore.Active) = throw(ArgumentError("Enzyme's `Active` type not supported in `Flux.gradient` or `Flux.withgradient`."))
_trymake_duplicated(x) = EnzymeCore.Duplicated(x, EnzymeCore.make_zero(x))


function _enzyme_gradient(f, args::Union{Const, Duplicated}...; zero::Bool=true)
    for x in args
        zero && x isa Duplicated && EnzymeCore.remake_zero!(x.dval)
        _check_mutable(x)
    end
    ad = Enzyme.set_runtime_activity(Reverse)
    Enzyme.autodiff(ad, Const(f), Active, args...)
    return map(_grad_or_nothing, args)
end

_check_mutable(x::Const) = nothing
_check_mutable(x::Duplicated) = Functors.anymutable(x) || error(
    """`Flux.gradient(f, Duplicated(x), ...)` expects `x` to contain mutable parameter arrays."""
)

# This function strips the returned gradient to be Zygote-like:
_grad_or_nothing(dup::Duplicated) = Flux.fmapstructure(_grad_or_nothing, dup.dval; prune=nothing)
_grad_or_nothing(::Const) = nothing
_grad_or_nothing(x) = Optimisers.isnumeric(x) ? x : nothing

function _enzyme_withgradient(f, args::Union{Const, Duplicated}...; zero::Bool=true)
    for x in args
        zero && x isa Duplicated && EnzymeCore.remake_zero!(x.dval)
        _check_mutable(x)
    end

    # Enzyme's reverse mode differentiates a single scalar (`Active`) output, so it can't natively
    # return auxiliary outputs the way Zygote's `withgradient` does. We adopt the trick Lux uses in
    # its training loop: wrap `f` so Enzyme only ever sees the scalar loss, and smuggle any auxiliary
    # outputs out through the (non-differentiated) wrapper as a side effect, to be read back after
    # `autodiff`. `_WithAux` also avoids reconstructing the auxiliary container *inside* the
    # differentiated call, which trips Enzyme (see its definition).
    wrapped = _WithAux(f)
    ad = Enzyme.set_runtime_activity(ReverseWithPrimal)
    _, loss = Enzyme.autodiff(ad, Const(wrapped), Active, args...)

    return (; val = _aux_val(wrapped, loss), grad = map(_grad_or_nothing, args))
end

# Wraps a loss `f` that may return a `Tuple`/`NamedTuple` whose first element is the scalar loss.
# The wrapper returns only that scalar to Enzyme (so reverse mode is happy) and stashes the
# auxiliary outputs in `aux` as a side effect. How the aux is stashed matters, because rebuilding it
# inside the differentiated call misbehaves with Enzyme:
#   * a `Tuple` return stashes `Base.tail` — storing the whole tuple instead makes Enzyme drop the
#     gradient to zero when the aux holds numeric arrays;
#   * a `NamedTuple` return stashes the whole value — rebuilding a `NamedTuple` tail inside `autodiff`
#     fails to compile for non-trivial aux.
mutable struct _WithAux{F}
    f::F
    kind::Symbol   # :scalar | :tuple | :named
    aux::Any       # Base.tail(val) for :tuple, the whole val for :named, unused for :scalar
end
_WithAux(f) = _WithAux(f, :scalar, nothing)

function (w::_WithAux)(args...)
    val = w.f(args...)
    if val isa Tuple
        w.kind = :tuple
        w.aux = Base.tail(val)
        return val[1]
    elseif val isa NamedTuple
        w.kind = :named
        w.aux = val
        return val[1]
    else
        return val
    end
end

# Reassemble the full `val` from the primal `loss` returned by Enzyme and the stashed aux.
_aux_val(w::_WithAux, loss) =
    w.kind === :tuple ? (loss, w.aux...) :
    w.kind === :named ? w.aux :
    loss


end # FluxEnzymeExt
