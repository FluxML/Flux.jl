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

function Flux._enzyme_gradient(f, args::Union{Const, Duplicated}...; zero::Bool=true)
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

function Flux._enzyme_withgradient(f, args::Union{Const, Duplicated}...; zero::Bool=true)
  for x in args
    zero && x isa Duplicated && EnzymeCore.remake_zero!(x.dval)
    _check_mutable(x)
  end

  # In order to support auxillary outputs, we try different ways.

  # Take I, doesn't allow for aux at all.
  # _, val = Enzyme.autodiff(ReverseWithPrimal, f, Active, args...)

  # Take II, using split mode.
  forward, reverse = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Active, map(typeof, args)...)
  tape, result, shadow_result  = forward(Const(f), args...)
  reverse(Const(f), args..., _sensitivity(result), tape)

  # Take III, it may be more efficient to have the function write the loss into Ref(0.0)?
  # dup_loss = DuplicatedNoNeed(Ref(0f0), Ref(1f0))
  # # result = autodiff(Reverse, Const(_ref_loss!), Const, dup_loss, Const(f), args...)
  # _, result = autodiff(ReverseWithPrimal, Const(_ref_loss!), Const, dup_loss, Const(f), args...)

  return (; val = result, grad = map(_grad_or_nothing, args))
end

@inline _sensitivity(y::Real) = one(y)
@inline _sensitivity(ys::Tuple{Real,Vararg}) = (one(ys[1]), Enzyme.make_zero(Base.tail(ys))...)
@inline _sensitivity(ys::NamedTuple{S, <:Tuple{Real,Vararg}}) where S = NamedTuple{S}(_sensitivity(Tuple(ys)))
_sensitivity(y) = error("""`Flux.withgradient(f, xs...)` expects that `y = f(xs...)` is a real numnber,
    or else a Tuple or NamedTuple whose first element is a real number.""")

# function _ref_loss!(out::Ref, f, args...)  # for Take III above
#   val = f(args...)
#   out[] = _get_loss(val)  # saves loss by mutation
#   val  # returns the whole thing
# end

# @inline _get_loss(y::Real) = y
# @inline _get_loss(ys::Tuple{Real,Vararg}) = ys[1]
# @inline _get_loss(ys::NamedTuple{S, <:Tuple{Real,Vararg}}) where S = ys[1]
# _get_loss(y) = error("""`Flux.withgradient(f, xs...)` expects that `y = f(xs...)` is a real numnber,
#     or else a Tuple or NamedTuple whose first element is a real number.""")

### Flux.Train, for train!

_applyloss(loss, model, d...) = loss(model, d...)

function _enzyme_train!(loss, model::Duplicated, data, opt; cb = nothing)
  isnothing(cb) || error("""train! does not support callback functions.
                            For more control use a loop with `gradient` and `update!`.""")
  @withprogress for (i,d) in enumerate(data)
    d_splat = d isa Tuple ? d : (d,)

    EnzymeCore.remake_zero!(model.dval)
    ad = Enzyme.set_runtime_activity(ReverseWithPrimal)
    _, l = Enzyme.autodiff(ad, _applyloss,
                           Active, Const(loss), model, map(Const, d_splat)...)

    if !isfinite(l)
      throw(DomainError(lazy"Loss is $l on data item $i, stopping training"))
    end
    opt, model2 = Optimisers.update!(opt, model.val, model.dval)
    model = Duplicated(model2, model.dval)

    @logprogress Base.haslength(data) ? i/length(data) : nothing
  end
end

end # FluxEnzymeExt
