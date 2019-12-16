"""
    Lookahead(opt, α = 0.5, k = 6, momentum_handler = default_momentum_handler!)

Implements the Lookahead optimiser.

## Parameters
  - Inner Optimiser
  - alpha (`\alpha::Float64`): The slow weights (outer loop) step size. Defaults to `0.5`.
  - k : The inner-loop step. The outer loop will update per k inner-loop step.
  - momentum_handler: function for handling the inner optimiser momentums. Default do nothing.

## Examples
```julia
inner_opt = ADAM() # create the inner optimiser

opt = Lookahead(inner_opt, α = 0.7, k = 10)
```

For using with custom Optimiser, overload the corresponding methods. see also [`has_momentum`](@ref), [`reset_momentum_handler!`], [`pullback_momentum_handler!`](@ref).

## References
[Lookahead](https://arxiv.org/pdf/1907.08610.pdf) optimiser.
"""
mutable struct Lookahead{O, MH <: Function}
  opt::O
  momentum_handler::MH
  alpha::Float64
  k::Int
  state::IdDict
end

const MomentumOptim = Union{Momentum, RMSProp, Nesterov, ADAM, RADAM, AdaMax, ADAGrad, ADADelta, AMSGrad, NADAM}

"""
  has_momentum(o)

return `true` if the optimiser has momentum.
"""
has_momentum(o) = false
has_momentum(o::MomentumOptim) = true
has_momentum(o::Optimiser) = any(has_momentum, o.os)
has_momentum(o::Lookahead) = has_momentum(o.opt)

_reset!(x::AbstractArray{T}) where T = (x .= zero(T))

@inline get_state(o::Union{Momentum, Nesterov}) = o.velocity
@inline get_state(o::Union{RMSProp, ADAGrad}) = o.acc
@inline get_state(o::Union{ADAM, RADAM, AdaMax, ADADelta, AMSGrad, NADAM}) = o.state
@inline get_state(o, x) = get_state(o)[x]

@inline _new_state!(o::Union{Momentum, Nesterov, RMSProp}, x) = _reset!(get_state(o, x))
@inline _new_state!(o::Union{ADAM, AdaMax, NADAM}, x) = (s = get_state(o, x); (_reset!(s[1]), _reset!(s[2]), o.beta))
@inline _new_state!(o::RADAM, x) = (s = get_state(o, x); (_reset!(s[1]), _reset!(s[2]), o.beta, 1))
@inline _new_state!(o::ADAGrad, x) = fill!(get_state(o, x), ϵ)
@inline _new_state!(o::ADADelta, x) = _reset!.(get_state(o, x))
@inline _new_state!(o::AMSGrad, x) = fill!.(get_state(o, x), ϵ)

reset_state!(o, x) = (get_state(o)[x] = _new_state!(o, x); nothing)
function reset_state!(o::Optimiser, x)
  for oi in o
    if has_momentum(oi)
      reset_state!(oi, x)
    end
  end
  return nothing
end

# momentum handlers
"do nothing"
default_momentum_handler!(o::Lookahead, x) = nothing

"Reset the inner optimiser momentums. Need to overload `reset_state!` for custom optimiser."
reset_momentum_handler!(o::Lookahead, x) = reset_state!(o.opt, x)

"return the momentum arrays in `Tuple`"
@inline momentum_buffer(o, x) = (s = get_state(o, x); s isa Tuple ? s : (s,))
@inline momentum_buffer(o::Union{ADAM, RADAM, AdaMax, NADAM}, x) = get_state(o, x)[1:2]
@inline function momentum_buffer(o::Optimiser, x)
  mapreduce(Base.Fix2(momentum_buffer, x), (init, x)->(init..., x...), filter(has_momentum, o); init=());
end

"pullback the inner momentum with outer momentum. Need to overload `momentum_buffer` for custom optimiser."
function pullback_momentum_handler!(o::Lookahead, x)
  opt, α = o.opt, o.alpha
  inter_mom = momentum_buffer(opt, x)
  outer_mom = map(inter_mom) do mom
    get!(o.state, mom, mom)
  end
  map(zip(inter_mom, outer_mom)) do (fast_mom, slow_mom)
    @. fast_mom = slow_mom = α * fast_mom + (1 - α) * slow_mom
  end
  return
end

"return `true` if the handler need optimiser has momentum."
_require_momentum(::Function) = true
_require_momentum(::typeof(default_momentum_handler!)) = false

function Lookahead(opt, α = 0.5, k = 6, momentum_handler = default_momentum_handler!)
  _require_momentum(momentum_handler) && !has_momentum(opt) && error("Inner optimizer $opt does not have momentum")
  Lookahead{typeof(opt), typeof(momentum_handler)}(opt, momentum_handler, α, k, IdDict())
end

function apply!(o::Lookahead{O}, x, Δ) where O
  slow_x, t = get!(o.state, x, (similar(x) .= x, 1))
  apply!(o.opt, x, Δ)
  if t >= o.k
    t = 0
    α = o.alpha
    fast_x = x .- Δ
    @. slow_x = α * fast_x + (1 - α) * slow_x
    @. Δ = x - slow_x
    o.momentum_handler(o, x)
  end
  o.state[x] = (slow_x, t+1)
  return Δ
end
