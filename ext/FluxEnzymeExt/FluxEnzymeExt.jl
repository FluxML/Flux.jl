module FluxEnzymeExt

using Flux
import Flux.Train: train!, _rule_to_state
import Flux.Optimise
import Optimisers
import Enzyme
using Enzyme: EnzymeRules, Active, Const, Duplicated, autodiff, ReverseWithPrimal
using ProgressLogging: @withprogress, @logprogress

_make_zero_internal!(x::AbstractArray) = fill!(x, 0)
_make_zero_internal!(x) = x
_make_zero!(model) = fmap(_make_zero_internal!, model)

_applyloss(loss, model, d...) = loss(model, d...)

EnzymeRules.inactive(::typeof(Flux.Losses._check_sizes), args...) = true

function train!(loss, model::Duplicated, data, rule::Optimisers.AbstractRule; cb = nothing)
  train!(loss, model, data, _rule_to_state(model, rule); cb)
end

function train!(loss, model::Duplicated, data, opt; cb = nothing)
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