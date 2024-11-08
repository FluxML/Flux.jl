module FluxEnzymeExt

using Flux
using Flux: _make_zero!

import Flux.Train: _enzyme_train!, _rule_to_state, _grad_or_nothing
# import Flux.Optimise

import Optimisers
import Enzyme
using Enzyme: EnzymeRules, Active, Const, Duplicated, autodiff, ReverseWithPrimal
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
  # TODO allow for f to return a tuple here, like in Zygote
  _, val = Enzyme.autodiff(ReverseWithPrimal, f, Active, args...)
  (; val, grad = map(_grad_or_nothing, args))
end

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


### Optimisers.update!, piracy, for now!

"""
    Flux.update!(opt_state, model::Duplicated)

Method of `update!` for use with Enzyme, and in particular with `gradient(loss, Duplicated(model))`.
Since `Duplicated(model)` stores the gradient, `update!` can read it & update the model itself,
by calling `Flux.update!(opt_state, model.val, model.dval)`.

!!! warning "Experimental"
    Enzyme support like this is new and somewhat experimental.
    This method is piracy, and must either move to Optimisers.jl
    or else Flux should own this function, and fall back to Optimisers.
"""
function Flux.update!(opt_state, model::Duplicated)
  Flux.update!(opt_state, model.val, _grad_or_nothing(model))
  model
end

end # FluxEnzymeExt
