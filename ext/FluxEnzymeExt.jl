module FluxEnzymeExt

using Flux
import Flux.Train: _enzyme_train!

import Optimisers
import Functors
import Enzyme
using Enzyme: EnzymeCore, EnzymeRules, Active, Const, Duplicated, autodiff, ReverseWithPrimal, DuplicatedNoNeed
using Enzyme: autodiff_thunk, Reverse, ReverseSplitWithPrimal
using ProgressLogging: @withprogress, @logprogress

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

  # In order to support auxillary outputs, we try different ways.

  ## Take I, doesn't allow for aux at all.
  ad = Enzyme.set_runtime_activity(ReverseWithPrimal)
  _, result = Enzyme.autodiff(ReverseWithPrimal, Const(f), Active, args...)

  ## Take II, using split mode.
  ## This fails with RNNs https://github.com/EnzymeAD/Enzyme.jl/issues/2897
  # forward, reverse = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Active, map(typeof, args)...)
  # tape, result, shadow_result  = forward(Const(f), args...)
  # reverse(Const(f), args..., _sensitivity(result), tape)

  ## Take III, it may be more efficient to have the function write the loss into Ref(0.0)?
  ## This doesn't work with Reactant
  # dup_loss = DuplicatedNoNeed(Ref(0f0), Ref(1f0))
  # ad = Enzyme.set_runtime_activity(ReverseWithPrimal)
  # _, result = autodiff(ad, Const(_ref_loss!), Const, dup_loss, Const(f), args...)

  return (; val = result, grad = map(_grad_or_nothing, args))
end

## for Take II above
# @inline _sensitivity(y::Real) = one(y)
# @inline _sensitivity(ys::Tuple{Real,Vararg}) = (one(ys[1]), Enzyme.make_zero(Base.tail(ys))...)
# @inline _sensitivity(ys::NamedTuple{S, <:Tuple{Real,Vararg}}) where S = NamedTuple{S}(_sensitivity(Tuple(ys)))
# _sensitivity(y) = error("""`Flux.withgradient(f, xs...)` expects that `y = f(xs...)` is a real numnber,
#     or else a Tuple or NamedTuple whose first element is a real number.""")

# for Take III above
# function _ref_loss!(out::Ref, f, args...)  
#   val = f(args...)
#   out[] = _get_loss(val)  # saves loss by mutation
#   val  # returns the whole thing
# end
# @inline _get_loss(y::Number) = y
# @inline _get_loss(ys::Tuple{Number,Vararg}) = ys[1]
# @inline _get_loss(ys::NamedTuple{S, <:Tuple{Number,Vararg}}) where S = ys[1]
# _get_loss(y) = error("""`Flux.withgradient(f, xs...)` expects that `y = f(xs...)` is a real numnber,
#     or else a Tuple or NamedTuple whose first element is a real number.""")


### Flux.Train, for train!

function _enzyme_train!(loss, model::Duplicated, data, opt; cb = nothing)
  isnothing(cb) || error("""train! does not support callback functions.
                            For more control use a loop with `gradient` and `update!`.""")
  @withprogress for (i,d) in enumerate(data)
    d_splat = d isa Tuple ? d : (d,)
    l, gs = Flux.withgradient(loss, AutoEnzyme(), model, map(Const, d_splat)...)
    if !isfinite(l)
      throw(DomainError(lazy"Loss is $l on data item $i, stopping training"))
    end
    opt, model2 = Optimisers.update!(opt, model.val, model.dval)
    model = Duplicated(model2, model.dval)

    @logprogress Base.haslength(data) ? i/length(data) : nothing
  end
end

end # FluxEnzymeExt
