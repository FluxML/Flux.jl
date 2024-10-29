module FluxEnzymeExt

using Flux
using Flux: _make_zero!
import Flux.Train: train!, _rule_to_state
import Optimisers
import Enzyme
using Enzyme: EnzymeRules, Active, Const, Duplicated, autodiff, ReverseWithPrimal
using ProgressLogging: @withprogress, @logprogress

EnzymeRules.inactive(::typeof(Flux.Losses._check_sizes), args...) = true

### gradient & withgradient

"""
    gradient(f, args::Union{Const,Duplicated}...)

This should return the same answer as `gradient(f, args...)`,
but it uses Enzyme.jl instead of Zygote.jl to compute the derivative.

Only available when Enzyme is loaded!

Besides returning the gradient, this is also stored within the `Duplicated` object.
Calling `Enzyme.Duplicated(model)` allocates space for the gradient,
which is zero'd befor use when calling `gradient`.
With the keyword `zero=false`, the new gradient will instead be added to what is already stored.

!!! warning "Experimental"
    Enzyme support like this is new and somewhat experimental.

# Example
```
julia> using Flux

julia> model = Chain(Dense([3.0;;]));

julia> Flux.gradient(model, [1]) do m, x  # computed using Zygote
         sum(abs2, m(x))
       end
((layers = ((weight = [6.0;;], bias = [6.0], σ = nothing),),), [18.0])

julia> using Enzyme

julia> dup_model = Duplicated(model);  # allocates space for gradient

julia> Flux.gradient(dup_model, Const([1])) do m, x  # Enzyme, returns the same
         sum(abs2, m(x))
       end
((layers = ((weight = [6.0;;], bias = [6.0], σ = nothing),),), nothing)

julia> dup_model  # same gradient is also stored within Duplicated
Duplicated(
  Chain(
    Dense(1 => 1),                      # 2 parameters
  ),
  # norm(∇) ≈ 8.49
)

julia> Flux.destructure((weight = [6.0;;], bias = [6.0]))[1] |> norm
8.48528137423857
```
"""
function Flux.gradient(f, args::Union{Const, Duplicated}...; zero::Bool=true)
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

# _const_unless_dup(x) = Const(x)
# _const_unless_dup(dup::Duplicated) = x

# TODO allow for Duplicated as 2nd argument, assume others const? This produces ambiguities...
# Flux.withgradient(f, dup::Duplicated, rest...) = Flux.withgradient(f, dup, map(_const_unless_dup, rest)...)
# Flux.gradient(f, dup::Duplicated, rest...) = Flux.gradient(f, dup, map(_const_unless_dup, rest)...)

"""
    withgradient(f, args::Union{Const,Duplicated}...)

This should return the same answer as `withgradient(f, model, args...)`,
but it uses Enzyme.jl instead of Zygote.jl to compute the derivative.

Only available when Enzyme is loaded!

Does not at present allow `f` to return a tuple of `(loss, aux)` the way `Zygote.withgradient` does.

# Example

```
julia> using Flux, Enzyme

julia> model = Chain(Embedding([1.1 2.2 3.3]), Dense([4.4;;]), only);

julia> model(3)
14.52

julia> Flux.withgradient(m -> m(3), model)  # this uses Zygote
(val = 14.52, grad = ((layers = ((weight = [0.0 0.0 4.4],), (weight = [3.3;;], bias = [1.0], σ = nothing), nothing),),))

julia> Flux.withgradient(m -> m(3), Duplicated(model))  # this uses Enzyme
(val = 14.52, grad = ((layers = ((weight = [0.0 0.0 4.4],), (weight = [3.3;;], bias = [1.0], σ = nothing), nothing),),))
```
"""
function Flux.withgradient(f, args::Union{Const, Duplicated}...; zero::Bool=true)
  for x in args
    zero && x isa Duplicated && _make_zero!(x.dval)
  end
  # TODO allow for f to return a tuple here, like in Zygote
  _, val = Enzyme.autodiff(ReverseWithPrimal, f, Active, args...)
  (; val, grad = map(_grad_or_nothing, args))
end

### Flux.Train, for train!

_applyloss(loss, model, d...) = loss(model, d...)

using Flux: _old_to_new  # from src/deprecations.jl
train!(loss, model::Duplicated, data, opt::Optimise.AbstractOptimiser; cb=nothing) =
  train!(loss, model, data, _old_to_new(opt); cb)

function train!(loss, model::Duplicated, data, rule::Optimisers.AbstractRule; cb = nothing)
  train!(loss, model, data, _rule_to_state(model, rule); cb)
end

"""
    train!(loss, Duplicated(model), data, opt_state)

This method uses Enzyme.jl instead of Zygote.jl to compute the gradients,
but is otherwise the same as `train!(loss, model, data, opt_state)`.

Only available when Enzyme is loaded.
"""
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
    This method is piracy, and must either move to Optimisers.jl
    or else Flux should own this function, and fall back to Optimisers.
"""
function Flux.update!(opt_state, model::Duplicated)
  Flux.update!(opt_state, model.val, _grad_or_nothing(model))
  model
end

end # FluxEnzymeExt
