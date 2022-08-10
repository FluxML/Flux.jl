module Train

using LinearAlgebra
using Optimisers: Optimisers
using Functors: fmap

export train!, update!, adjust!, FluxState,
	Descent, Adam, Momentum, Nesterov, RMSProp,
	AdaGrad, AdaMax, AdaDelta, AMSGrad, NAdam, AdamW, RAdam, OAdam, AdaBelief #,
  # InvDecay, ExpDecay, WeightDecay, stop, skip, Optimiser,
  # ClipValue, ClipNorm


### Mutable state storage, to wrap Optimisers.jl
  
"""
    FluxState(rule, state=missing)

This is an interface between the all-mutable world Flux.jl likes,
and the could-be-immutable world that Optimisers.jl inhabits.

`state` can can be either the whole state tree which Optimisers.jl builds,
or else (for Zygote's implicit mode) an IdDict of such states.
Once initialised, it cannot change between these two modes.
"""
mutable struct FluxState{T<:Optimisers.AbstractRule};
  rule::T
  state::Any
end

function Base.show(io::IO, opt::FluxState)
  print(io, "FluxState(")
  show(io, opt.rule)
  if opt.state isa Missing
    print(io, ", <uninitialised>)")
  elseif opt.state isa IdDict
    n = length(keys(opt.state))
    print(io, ", <implicit IdDict: $n arrays>))")
  else
    rn = Ref(0)
    fmap(x -> (rn[]+=1; x), opt.state, exclude = (x -> x isa Optimisers.Leaf))
    print(io, ", <explicit tree: $(rn[]) leaves>)")
  end
end

_DESCENT_EXAMPLE = """# Implicit-style example
This usage matches Flux ≤ v0.13:
```
opt = Flux.Descent(0.3)

ps = Flux.params(model)  # returns a Zygote.Params object

gs = gradient(ps) do    # gradient takes a zero-argument anonymous function
  loss3(model, x, y)    # ... which depends on the global model
end                     # ... and returns a Zygote.Grads object

Flux.update!(opt, ps, gs)
```
New on Flux v0.14 is a method `train!(loss, ps, opt)` which performs one step,
rather than iterating over `data`. This is equivalent to `gradient` and `update!` above:
```
Flux.train!(ps, opt) do
  loss3(model, x, y)
end
```

# Explicit-style example

This no longer uses `Flux.params`, but instead the model itself:
```
opt = Flux.Descent(0.3)        # the same FluxState object

Flux.train!(model, opt) do m   # now explicitly depends on the model
  loss3(m, x, y)
end
```
"""
for opt in [
  :Descent, :Adam, :Momentum, :Nesterov, :RMSProp,
	:AdaGrad, :AdaMax, :AdaDelta, :AMSGrad, :NAdam, :AdamW, :RAdam, :OAdam, :AdaBelief,
	# :InvDecay, :ExpDecay, :WeightDecay, :Optimiser,
  :ClipGrad, :ClipNorm,
# TODO sort out the remaining rules
]
  @eval begin 
    $opt(parameters...; kw...) = FluxState(Optimisers.$opt(parameters...; kw...), missing)
    str = string("""    Flux.$($opt)(args...)
    
    Returns `FluxState` wrapper around the following rule definition from Optimisers.jl,
    allowing its use with `Flux.train!` (in the same manner as `Flux.AbstractOptimiser` objects on Flux ≤ v0.13).
    Accepts the same arguments, with the same defaults, as the underlying rule:
    
    """, @doc(Optimisers.$opt), $opt == Descent ? _DESCENT_EXAMPLE : "")
    @doc str $opt
  end
end

@deprecate ClipValue ClipGrad


### Two styles of gradient, and their `train!` functions

using ProgressLogging: @progress, @withprogress, @logprogress  # TODO add progress logging again
using Zygote: Zygote, Params

include("explicit_train.jl")  # new!
include("implicit_train.jl")  # Params etc, Zygote only

explicit_withgradient(f, args...) = Zygote.withgradient(f, args...)  # can overload this to use e.g. Yota / Diffractor

### Misc. related utilities

"""
    Flux.adjust!(opt::FluxState, η::Real)

Alters the learning rate of the optimiser,
without resetting its stored momentum state, etc.
"""
function adjust!(opt::FluxState, eta::Real)
  opt.rule = Optimisers.adjust(opt.rule, eta)
  s = opt.state
  if s isa missing
  elseif s isa IdDict
    for k in keys(s)
      s[k] = Optimisers.adjust(s[k], eta)
    end
  else
    s = Optimisers.adjust(s, eta)
  end
  opt.state = s
  return opt
end

end # module
