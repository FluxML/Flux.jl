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
    Flux.update!(opt, model)

    @logprogress Base.haslength(data) ? i/length(data) : nothing
  end
end

end # FluxEnzymeExt
