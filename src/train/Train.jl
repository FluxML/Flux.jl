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

for opt in [
  :Descent, :Adam, :Momentum, :Nesterov, :RMSProp,
	:AdaGrad, :AdaMax, :AdaDelta, :AMSGrad, :NAdam, :AdamW, :RAdam, :OAdam, :AdaBelief,
	# :InvDecay, :ExpDecay, :WeightDecay, :stop, :skip, :Optimiser,
  # :ClipValue, :ClipNorm,
# TODO check that parameters line up nicely old-vs-new, and include the remaining rules
]
  @eval $opt(parameters...; kw...) = FluxState(Optimisers.$opt(parameters...; kw...), missing)
end


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
