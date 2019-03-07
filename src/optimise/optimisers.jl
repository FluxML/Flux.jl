using Flux
using Base: @get!
using MacroTools: @forward

const ϵ = 1e-8

# TODO: should use weak refs

"""
    Descent(η)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `δp`, this runs `p -= η*δp`.
"""
mutable struct Descent
  eta::Float64
end

Descent() = Descent(0.1)

function apply!(o::Descent, x, Δ)
  Δ .*= o.eta
end

"""
    Momentum(params, η = 0.01; ρ = 0.9)

Gradient descent with learning rate `η` and momentum `ρ`.
"""
mutable struct Momentum
  eta::Float64
  rho::Float64
  velocity::IdDict
end

Momentum(η = 0.01, ρ = 0.9) = Momentum(η, ρ, IdDict())

function apply!(o::Momentum, x, Δ)
  η, ρ = o.eta, o.rho
  v = get!(o.velocity, x, zero(x))::typeof(data(x))
  @. v = ρ * v - η * Δ
  @. Δ = -v
end

"""
    Nesterov(eta, ρ = 0.9)

Gradient descent with learning rate  `η` and Nesterov momentum `ρ`.
"""
mutable struct Nesterov
  eta::Float64
  rho::Float64
  velocity::IdDict
end

Nesterov(η = 0.001, ρ = 0.9) = Nesterov(η, ρ, IdDict())

function apply!(o::Nesterov, x, Δ)
  η, ρ = o.eta, o.rho
  v = get!(o.velocity, x, zero(x))::typeof(data(x))
  d = @. ρ^2 * v - (1+ρ) * η * Δ
  @. v = ρ*v - η*Δ
  @. Δ = -d
end

"""
    RMSProp(η = 0.001, ρ = 0.9)

[RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
optimiser. Parameters other than learning rate don't need tuning. Often a good
choice for recurrent networks.
"""
mutable struct RMSProp
  eta::Float64
  rho::Float64
  acc::IdDict
end

RMSProp(η = 0.001, ρ = 0.9) = RMSProp(η, ρ, IdDict())

function apply!(o::RMSProp, x, Δ)
  η, ρ = o.eta, o.rho
  acc = get!(o.acc, x, zero(x))::typeof(data(x))
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= η / (√acc + ϵ)
end

"""
    ADAM(η = 0.001, β = (0.9, 0.999))

[ADAM](https://arxiv.org/abs/1412.6980v8) optimiser.
"""
mutable struct ADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
end

ADAM(η = 0.001, β = (0.9, 0.999)) = ADAM(η, β, IdDict())

function apply!(o::ADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  o.state[x] = (mt, vt, βp .* β)
  return Δ
end

"""
    AdaMax(params, η = 0.001; β1 = 0.9, β2 = 0.999, ϵ = 1e-08)

[AdaMax](https://arxiv.org/abs/1412.6980v9) optimiser. Variant of ADAM based on
the ∞-norm.
"""
mutable struct AdaMax
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
end

AdaMax(η = 0.001, β = (0.9, 0.999)) = AdaMax(η, β, IdDict())

function apply!(o::AdaMax, x, Δ)
  η, β = o.eta, o.beta
  mt, ut, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. ut = max(β[2] * ut, abs(Δ))
  @. Δ = (η/(1 - βp[1])) * mt/(ut + ϵ)
  o.state[x] = (mt, ut, βp .* β)
  return Δ
end

"""
    ADAGrad(η = 0.1; ϵ = 1e-8)

[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimiser.
Parameters don't need tuning.
"""
mutable struct ADAGrad
  eta::Float64
  acc::IdDict
end

ADAGrad(η = 0.1) = ADAGrad(η, IdDict())

function apply!(o::ADAGrad, x, Δ)
  η = o.eta
  acc = get!(o.acc, x, fill(ϵ, size(x)))::typeof(data(x))
  @. acc += Δ^2
  @. Δ *= η / (√acc + ϵ)
end

"""
    ADADelta(ρ = 0.9, ϵ = 1e-8)

[ADADelta](http://arxiv.org/abs/1212.5701) optimiser. Parameters don't need
tuning.
"""
mutable struct ADADelta
  rho::Float64
  state::IdDict
end

ADADelta(ρ = 0.9) = ADADelta(ρ, IdDict())

function apply!(o::ADADelta, x, Δ)
  ρ = o.rho
  acc, Δacc = get!(o.state, x, (zero(x), zero(x)))
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= √Δacc/ (√acc + ϵ)
  @. Δacc = ρ * Δacc + (1 - ρ) * Δ^2
  return Δ
end

"""
    AMSGrad(η = 0.001, β = (0.9, 0.999))

[AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) optimiser. Parameters don't need
tuning.
"""
mutable struct AMSGrad
  eta::Float64
  beta::Tuple{Float64, Float64}
  state::IdDict
end

AMSGrad(η = 0.001, β = (0.9, 0.999)) = AMSGrad(η, β, IdDict())

function apply!(o::AMSGrad, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, v̂t = get!(o.state, x, (fill(ϵ, size(x)), fill(ϵ, size(x)), fill(ϵ, size(x))))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ ^ 2
  @. v̂t = max.(v̂t, vt)
  @. Δ = η * mt / (√v̂t + ϵ)
end

"""
    NADAM(η = 0.001, β = (0.9, 0.999))

[NADAM](http://cs229.stanford.edu/proj2015/054_report.pdf) optimiser. Parameters don't need
tuning.
"""
mutable struct NADAM
  eta::Float64
  beta::Tuple{Float64, Float64}
  state::IdDict
end

NADAM(η = 0.001, β = (0.9, 0.999)) = NADAM(η, β, IdDict())

function apply!(o::NADAM, x, Δ)
  η, β = o.eta, o.beta
  β1p, β2p = o.beta
  mt, vt = get!(o.state, x, (zero(x), zero(x)))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ = (β[1] * mt / (1 - β[1] * β1p) + (1 - β[1]) * Δ / (1 - β1p)) / (√(vt * β[2] / (1 - β2p)) + ϵ) * η
  o.state[x] = (mt, vt, (β1p * β[1], β2p * β[2]))
  return Δ
end

"""
    ADAMW((η = 0.001, β = (0.9, 0.999), decay = 0)

[ADAMW](https://arxiv.org/abs/1711.05101) fixing weight decay regularization in Adam.
"""
ADAMW(η = 0.001, β = (0.9, 0.999), decay = 0) =
  Optimiser(ADAM(η, β), WeightDecay(decay))

# Compose optimizers

"""
    Optimiser(a, b, c...)

Combine several optimisers into one; each optimiser produces a modified gradient
that will be fed into the next, and this is finally applied to the parameter as
usual.
"""
mutable struct Optimiser
  os::Vector{Any}
end

Optimiser(o...) = Optimiser(Any[o...])

@forward Optimiser.os Base.getindex, Base.first, Base.last, Base.lastindex, Base.push!, Base.setindex!
@forward Optimiser.os Base.iterate

Base.getindex(c::Optimiser, i::AbstractArray) = Optimiser(c.os[i]...)

function apply!(o::Optimiser, x, Δ)
  for opt in o.os
    Δ = apply!(opt, x, Δ)
  end
  return Δ
end

"""
`InvDecay(γ)`

Apply inverse time decay to an optimiser
```julia
  Optimiser(InvDecay(..), Opt(..))
```
"""
mutable struct InvDecay
  gamma::Float64
  state::IdDict
end

InvDecay(γ = 0.001) = InvDecay(γ, IdDict())

function apply!(o::InvDecay, x, Δ)
  γ = o.gamma
  n = get!(o.state, x, 1)
  Δ .*= 1 / (1 + γ * n)
  o.state[x] = n + 1
  return Δ
end

"""
`ExpDecay(eta, decay, decay_step, clip)`

Schedule the learning rate `eta` by `decay` every `decay_step` till a minimum of `clip`.

To apply exponential decay to an optimiser:
```julia
  Optimiser(ExpDecay(..), Opt(..))
```
"""
mutable struct ExpDecay
  eta::Float64
  decay::Float64
  step::Int64
  clip::Float64
  current::IdDict
end

ExpDecay(opt = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4) = ExpDecay(opt, decay, decay_step, clip, IdDict())

function apply!(o::ExpDecay, x, Δ)
  η, s, decay = o.eta, o.step, o.decay
  n = o.current[x] = get(o.current, x, 0) + 1
  if o.current[x]%s == 0 && count(x -> x%s == 0, values(o.current)) == 1
    η = max(η * decay^(s / n), o.clip)
    o.eta = η
  end
  @. Δ *= decay
end

"""
`WeightDecay(wd)`

Decay the weight parameter by `wd`
"""
mutable struct WeightDecay
  wd::Real
end

WeightDecay() = WeightDecay(0)

function apply!(o::WeightDecay, x, Δ)
  wd = o.wd
  @. Δ += wd * data(x)
end
