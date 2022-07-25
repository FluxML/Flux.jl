module Train

using LinearAlgebra
using Optimisers: Optimisers
using Functors: fmap

export train!, update!, adjust!, FluxState, @epochs,
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

using ProgressLogging: @progress, @withprogress, @logprogress
using Zygote: Zygote, Params

include("explicit_train.jl.jl")  # new!
include("implicit_train.jl.jl")  # Params etc, Zygote only

explicit_withgradient(f, args...) = Zygote.withgradient(f, args...)  # can overload this to use e.g. Yota / Diffractor

# using Requires  # Flux doesn't use this right now
# @init @require Diffractor="9f5e2b26-1114-432f-b630-d3fe2085c51c" begin
#   @eval function explicit_withgradient(f, args...)
#     y, back = Diffractor.∂⃖¹(f, args...)
#     _, grads... = back(Zygote.sensitivity(y))
#     return (; value = y, gradient = grads)
#   end
# end

#=

using Diffractor
function Flux.Train.explicit_withgradient(f, args...)
  y, back = Diffractor.∂⃖¹(f, args...)
  _, grads... = back(one(y))
  return (; value = y, gradient = grads)
end

=#

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

"""
    @epochs N body

Run `body` expression `N` times. Mainly useful for quickly doing
multiple epochs of training in a REPL.

Functionally equivalent to this loop:
```
for _ in 1:N   
    body
end
```
... but adds progress logging and `@info` messages,
and returns the result of the last iteration.

# Examples
```jldoctest
julia> Flux.@epochs 2 println("hello")
[ Info: Epoch 1
hello
[ Info: Epoch 2
hello
```
"""
macro epochs(n, ex)
  @gensym val
  body = :(for i in 1:$(esc(n))
    @info "Epoch $i"
    $(esc(val)) = $(esc(ex))
  end)
  loop = Expr(:macrocall, Symbol("@progress"), __source__, body)
  Expr(:block, :($(esc(val)) = nothing), loop, :($(esc(val))))
  # TODO make this actualy return the value? Names aren't right.
#
#     $loop
#     # @progress for i in 1:$(esc(n))
# #         @info "Epoch $i"
# #         $(esc(val)) = $(esc(ex))
# #     end
#     $val  # DOESN"T WORK! Expr(:macrocall, ...) ?
#   end
end

end


#=

using Flux, Random
data = [(rand(3,2).*[i,1,20/i], [i i]) for i in 1:50] |> shuffle!;

# This exact code works on Flux@0.13. There, train! returns nothing:
model2 = Chain(Dense(3 => 7, relu), Dense(7 => 1))
opt2 = Flux.Adam()
Flux.train!(Flux.params(model2), data, opt2) do x, y
  Flux.mse(model2(x), y)
end
opt2  # contains an IdDict

# This is the new "explicit" method of Train
model1 = Chain(Dense(3 => 7, relu), Dense(7 => 1))
opt1 = Flux.Adam()
Flux.train!(model1, data, opt1) do m, x, y
  Flux.mse(m(x), y)
end |> sum
opt1  # contains state tree

# This is new 3-arg train!, one step not an iteration over data:
x1, y1 = data[1]
Flux.train!(model1, opt1) do m
  Flux.mse(m(x1), y1)
end





julia> using ProgressLogging
julia> @macroexpand1 @loop N body
begin
  x = nothing
  @progress for i in 1:N
    @info "step $i"
    x = body
  end
  x
end



=#