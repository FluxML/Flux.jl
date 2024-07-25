module FluxEnzymeExt

using Flux
import Flux.Train: train!, _rule_to_state
import Optimisers
import Enzyme
using Enzyme: EnzymeRules, Active, Const, Duplicated, autodiff, ReverseWithPrimal
using ProgressLogging: @withprogress, @logprogress

_make_zero_internal!(x::AbstractArray) = fill!(x, 0)
_make_zero_internal!(x) = x
_make_zero!(model) = fmap(_make_zero_internal!, model)

EnzymeRules.inactive(::typeof(Flux.Losses._check_sizes), args...) = true

### gradient & withgradient

_grad_or_nothing(dup::Duplicated) = Flux.fmapstructure(_grad_or_nothing, dup.dval)
_grad_or_nothing(::Const) = nothing
_grad_or_nothing(x) = Optimisers.isnumeric(x) ? x : nothing

function Flux.withgradient(f, args::Union{Const, Duplicated}...)
  for x in args
    x isa Duplicated && _make_zero!(x.dval)
  end
  # TODO allow for f to return a tuple here, like in Zygote
  _, val = Enzyme.autodiff(ReverseWithPrimal, f, Active, args...)
  (; val, grad = map(_grad_or_nothing, args))
end

"""
    gradient(f, Duplicated(model), args...)

This should return the same answer as `gradient(f, model, args...)`,
but it uses Enzyme.jl instead of Zygote.jl to compute the derivative.

Only available when Enzyme is loaded!

Besides returning the gradient, this is also stored within the `Duplicated` object.
Calling `Enzyme.Duplicated(model)` allocates space for the gradient,
which is zero'd befor use when calling `gradient`.

!!! warning "Experimental"
    Enzyme support like this is new and somewhat experimental.
    It has known problems if your model has shared parameters.

# Example
```
julia> using Flux, Enzyme

julia> model = Chain(Dense([3.0;;]));

julia> Flux.gradient(model, [1]) do m, x
         sum(abs2, m(x))
       end
((layers = ((weight = [6.0;;], bias = [6.0], σ = nothing),),), [18.0])

julia> Flux.gradient(Duplicated(model), Const([1])) do m, x
         sum(abs2, m(x))
       end
┌ Warning: Using fallback BLAS replacements for (["dsymv_64_"]), performance may be degraded
└ @ Enzyme.Compiler ~/.julia/packages/GPUCompiler/Y4hSX/src/utils.jl:59
((layers = ((weight = [6.0;;], bias = [6.0], σ = nothing),),), nothing)

```
"""
function Flux.gradient(f, args::Union{Const, Duplicated}...)
  for x in args
    x isa Duplicated && _make_zero!(x.dval)
  end
  Enzyme.autodiff(ReverseWithPrimal, f, Active, args...)
  map(_grad_or_nothing, args)
end

_const_unless_dup(x) = Const(x)
_const_unless_dup(dup::Duplicated) = x

# TODO allow for Duplicated as 2nd argument, assume others const? This produces ambiguities...
# Flux.withgradient(f, dup::Duplicated, rest...) = Flux.withgradient(f, dup, map(_const_unless_dup, rest)...)
# Flux.gradient(f, dup::Duplicated, rest...) = Flux.gradient(f, dup, map(_const_unless_dup, rest)...)


### Flux.Train, for train!

_applyloss(loss, model, d...) = loss(model, d...)

using Flux: _old_to_new  # from src/deprecations.jl
train!(loss, model::Duplicated, data, opt::Optimise.AbstractOptimiser; cb=nothing) =
  train!(loss, model, data, _old_to_new(opt); cb)

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


### Optimisers.update!, piracy, for now!

"""
    Flux.update!(opt_state, model::Duplicated)

Method of `update!` for use with Enzyme, and in particular with `gradient(loss, Duplicated(model))`.
Since `Duplicated(model)` stores the gradient, `update!` can read it & update the model itself,
by calling `Flux.update!(opt_state, model.val, model.dval)`.

!!! warning "Experimental"
    Enzyme support like this is new and somewhat experimental.
    This method is piracy, and must move to Optimisers.jl in the end.
"""
function Flux.update!(opt_state, model::Duplicated)
  Flux.update!(opt_state, model.val, model.dval)
  model
end

end # FluxEnzymeExt
