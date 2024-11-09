module FluxEnzymeExt

using Flux
using Flux: _make_zero!

import Flux.Train: _enzyme_train!, _rule_to_state, _grad_or_nothing
# import Flux.Optimise

import Optimisers
import Enzyme
using Enzyme: EnzymeRules, Active, Const, Duplicated, autodiff, ReverseWithPrimal
using Enzyme: autodiff_thunk, ReverseSplitWithPrimal
using ProgressLogging: @withprogress, @logprogress

EnzymeRules.inactive(::typeof(Flux.Losses._check_sizes), args...) = true

### gradient & withgradient

function Flux._enzyme_gradient(f, args::Union{Const, Duplicated}...; zero::Bool=true)
  for x in args
    zero && x isa Duplicated && _make_zero!(x.dval)
  end
  Enzyme.autodiff(ReverseWithPrimal, f, Active, args...)
  map(_grad_or_nothing, args)
end

# This function strips the returned gradient to be Zygote-like:
_grad_or_nothing(dup::Duplicated) = Flux.fmapstructure(_grad_or_nothing, dup.dval; prune=nothing)
_grad_or_nothing(::Const) = nothing
_grad_or_nothing(x) = Optimisers.isnumeric(x) ? x : nothing

function Flux._enzyme_withgradient(f, args::Union{Const, Duplicated}...; zero::Bool=true)
  for x in args
    zero && x isa Duplicated && _make_zero!(x.dval)
  end

  # _, val = Enzyme.autodiff(ReverseWithPrimal, f, Active, args...)

  forward, reverse = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Active, map(typeof, args)...)
  tape, result, shadow_result  = forward(Const(f), args...)
  reverse(Const(f), args..., _sensitivity(result), tape)

  (; val = result, grad = map(_grad_or_nothing, args))
end

_sensitivity(y::Real) = one(y)
_sensitivity(ys::Tuple{Real,Vararg}) = (one(ys[1]), Enzyme.make_zero(Base.tail(ys))...)
_sensitivity(ys::NamedTuple{S, <:Tuple{Real,Vararg}}) where S = NamedTuple{S}(_sensitivity(Tuple(ys)))
_sensitivity(y) = error("""`Flux.withgradient(f, xs...)` expects that `y = f(xs...)` is a real numnber,
    or else a Tuple or NamedTuple whose first element is a real number.""")

### Flux.Train, for train!

_applyloss(loss, model, d...) = loss(model, d...)

function _enzyme_train!(loss, model::Duplicated, data, opt; cb = nothing)
  isnothing(cb) || error("""train! does not support callback functions.
                            For more control use a loop with `gradient` and `update!`.""")
  @withprogress for (i,d) in enumerate(data)
    d_splat = d isa Tuple ? d : (d,)

    _make_zero!(model.dval)
    _, l = Enzyme.autodiff(ReverseWithPrimal, _applyloss,
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
