using Flux
using MacroTools: @forward

const ϵ = 1e-8

# TODO: should use weak refs

"""
    Descent(η = 0.1; η_dict = IdDict())

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `δp`, this runs `p -= η*δp`
Optionally, different learning rates can be specified for different parameters
through `η_dict`. The parameters without specific learning rates will use the
`η` value.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.

# Examples
```julia
opt = Descent()

opt = Descent(0.3)

ps = params(model)

# Use 0.3 η for all parameters except ps[2], use 0.2 for ps[2]
opt = Descent(0.3, IdDict([(ps[2], 0.2)]))

gs = gradient(ps) do
    loss(x, y)
end

Flux.Optimise.update!(opt, ps, gs)
```
"""
mutable struct Descent
  eta::Float64
  eta_dict::IdDict
end

Descent(η = 0.1; η_dict = IdDict()) = Descent(η, η_dict)

function apply!(o::Descent, x, Δ)
  η = get(o.eta_dict, x, o.eta)
  Δ .*= η
end

"""
    Momentum(η = 0.01, ρ = 0.9; η_dict = IdDict(), ρ_dict = IdDict())

Gradient descent optimizer with learning rate `η` and momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter momentum (`ρ_dict`): Same as the momentum but specific for each
                                     parameter.

# Examples
```julia
opt = Momentum()

opt = Momentum(0.01, 0.99)
```
"""
mutable struct Momentum
  eta::Float64
  rho::Float64
  velocity::IdDict
  eta_dict::IdDict
  rho_dict::IdDict
end

Momentum(η = 0.01, ρ = 0.9; η_dict = IdDict(), ρ_dict = IdDict()) = Momentum(η, ρ, IdDict(), η_dict, ρ_dict)

function apply!(o::Momentum, x, Δ)
  η, ρ = get(o.eta_dict, x, o.eta), get(o.rho_dict, x, o.rho)
  v = get!(o.velocity, x, zero(x))::typeof(x)
  @. v = ρ * v - η * Δ
  @. Δ = -v
end

"""
    Nesterov(η = 0.001, ρ = 0.9; η_dict = IdDict(), ρ_dict = IdDict())

Gradient descent optimizer with learning rate `η` and Nesterov momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Nesterov momentum (`ρ`): Controls the acceleration of gradient descent in the
                           prominent direction, in effect dampening oscillations.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter momentum (`ρ_dict`): Same as the momentum but specific for each
                                     parameter.

# Examples
```julia
opt = Nesterov()

opt = Nesterov(0.003, 0.95)
```
"""
mutable struct Nesterov
  eta::Float64
  rho::Float64
  velocity::IdDict
  eta_dict::IdDict
  rho_dict::IdDict
end

Nesterov(η = 0.001, ρ = 0.9; η_dict = IdDict(), ρ_dict = IdDict()) = Nesterov(η, ρ, IdDict(), η_dict, ρ_dict)

function apply!(o::Nesterov, x, Δ)
  η, ρ = get(o.eta_dict, x, o.eta), get(o.rho_dict, x, o.rho)
  v = get!(o.velocity, x, zero(x))::typeof(x)
  d = @. ρ^2 * v - (1+ρ) * η * Δ
  @. v = ρ*v - η*Δ
  @. Δ = -d
end

"""
    RMSProp(η = 0.001, ρ = 0.9; η_dict = IdDict(), ρ_dict = IdDict())

Optimizer using the
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
algorithm. Often a good choice for recurrent networks. Parameters other than learning rate
generally don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect dampening oscillations.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter momentum (`ρ_dict`): Same as the momentum but specific for each
                                     parameter.

# Examples
```julia
opt = RMSProp()

opt = RMSProp(0.002, 0.95)
```
"""
mutable struct RMSProp
  eta::Float64
  rho::Float64
  acc::IdDict
  eta_dict::IdDict
  rho_dict::IdDict
end

RMSProp(η = 0.001, ρ = 0.9; η_dict = IdDict(), ρ_dict = IdDict()) = RMSProp(η, ρ, IdDict(), η_dict, ρ_dict)

function apply!(o::RMSProp, x, Δ)
  η, ρ = get(o.eta_dict, x, o.eta), get(o.rho_dict, x, o.rho)
  acc = get!(o.acc, x, zero(x))::typeof(x)
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= η / (√acc + ϵ)
end

"""
    ADAM(η = 0.001, β::Tuple = (0.9, 0.999); η_dict = IdDict(), β_dict = IdDict())

[ADAM](https://arxiv.org/abs/1412.6980) optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter momentum decay (`β_dict`): Same as the decay of momentum but
                                           specific for each parameter.

# Examples
```julia
opt = ADAM()

opt = ADAM(0.001, (0.9, 0.8))
```
"""
mutable struct ADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
  eta_dict::IdDict
  beta_dict::IdDict
end

ADAM(η = 0.001, β = (0.9, 0.999); η_dict = IdDict(), β_dict = IdDict()) = ADAM(η, β, IdDict(), η_dict, β_dict)

function apply!(o::ADAM, x, Δ)
  η, β = get(o.eta_dict, x, o.eta), get(o.beta_dict, x, o.beta)
  mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  o.state[x] = (mt, vt, βp .* β)
  return Δ
end

"""
    RADAM(η = 0.001, β::Tuple = (0.9, 0.999); η_dict = IdDict(), β_dict = IdDict())

[Rectified ADAM](https://arxiv.org/abs/1908.03265) optimizer.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter momentum decay (`β_dict`): Same as the decay of momentum but
                                           specific for each parameter.

# Examples
```julia
opt = RADAM()

opt = RADAM(0.001, (0.9, 0.8))
```
"""
mutable struct RADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
  eta_dict::IdDict
  beta_dict::IdDict
end

RADAM(η = 0.001, β = (0.9, 0.999); η_dict = IdDict(), β_dict = IdDict()) = RADAM(η, β, IdDict(), η_dict, β_dict)

function apply!(o::RADAM, x, Δ)
  η, β = get(o.eta_dict, x, o.eta), get(o.beta_dict, x, o.beta)
  ρ∞ = 2/(1-β[2])-1
  mt, vt, βp, t = get!(o.state, x, (zero(x), zero(x), β, 1))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  ρ = ρ∞ - 2t*βp[2]/(1-βp[2])
  if ρ > 4
    r = sqrt((ρ-4)*(ρ-2)*ρ∞/((ρ∞-4)*(ρ∞-2)*ρ))
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η * r
  else
    @. Δ =  mt / (1 - βp[1]) * η
  end
  o.state[x] = (mt, vt, βp .* β, t+1)
  return Δ
end

"""
    AdaMax(η = 0.001, β::Tuple = (0.9, 0.999); η_dict = IdDict(), β_dict = IdDict())

[AdaMax](https://arxiv.org/abs/1412.6980) is a variant of ADAM based on the ∞-norm.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter momentum decay (`β_dict`): Same as the decay of momentum but
                                           specific for each parameter.

# Examples
```julia
opt = AdaMax()

opt = AdaMax(0.001, (0.9, 0.995))
```
"""
mutable struct AdaMax
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
  eta_dict::IdDict
  beta_dict::IdDict
end

AdaMax(η = 0.001, β = (0.9, 0.999); η_dict = IdDict(), β_dict = IdDict()) = AdaMax(η, β, IdDict(), η_dict, β_dict)

function apply!(o::AdaMax, x, Δ)
  η, β = get(o.eta_dict, x, o.eta), get(o.beta_dict, x, o.beta)
  mt, ut, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. ut = max(β[2] * ut, abs(Δ))
  @. Δ = (η/(1 - βp[1])) * mt/(ut + ϵ)
  o.state[x] = (mt, ut, βp .* β)
  return Δ
end

"""
    OADAM(η = 0.0001, β::Tuple = (0.5, 0.9); η_dict = IdDict(), β_dict = IdDict())

[OADAM](https://arxiv.org/abs/1711.00141) (Optimistic ADAM)
is a variant of ADAM adding an "optimistic" term suitable for adversarial training.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter momentum decay (`β_dict`): Same as the decay of momentum but
                                           specific for each parameter.

# Examples
```julia
opt = OADAM()

opt = OADAM(0.001, (0.9, 0.995))
```
"""
mutable struct OADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
  eta_dict::IdDict
  beta_dict::IdDict
end

OADAM(η = 0.0001, β = (0.5, 0.9); η_dict = IdDict(), β_dict = IdDict()) = OADAM(η, β, IdDict(), η_dict, β_dict)

function apply!(o::OADAM, x, Δ)
  η, β = get(o.eta_dict, x, o.eta), get(o.beta_dict, x, o.beta)
  mt, vt, Δ_, βp = get!(o.state, x, (zero(x), zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ = -Δ_
  @. Δ_ = η * mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ)
  @. Δ += 2Δ_
  o.state[x] = (mt, vt, Δ_, βp .* β)
  return Δ
end

"""
    ADAGrad(η = 0.1; η_dict = IdDict())

[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimizer. It has
parameter specific learning rates based on how frequently it is updated.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.

# Examples
```julia
opt = ADAGrad()

opt = ADAGrad(0.001)
```
"""
mutable struct ADAGrad
  eta::Float64
  acc::IdDict
  eta_dict::IdDict
end

ADAGrad(η = 0.1; η_dict = IdDict()) = ADAGrad(η, IdDict(), η_dict)

function apply!(o::ADAGrad, x, Δ)
  η = get(o.eta_dict, x, o.eta)
  acc = get!(o.acc, x, fill!(zero(x), ϵ))::typeof(x)
  @. acc += Δ^2
  @. Δ *= η / (√acc + ϵ)
end

"""
    ADADelta(ρ = 0.9; ρ_dict = IdDict())

[ADADelta](https://arxiv.org/abs/1212.5701) is a version of ADAGrad adapting its learning
rate based on a window of past gradient updates.
Parameters don't need tuning.

# Parameters
- Rho (`ρ`): Factor by which the gradient is decayed at each time step.
- Per-parameter momentum decay (`ρ_dict`): Same as `ρ` but specific for each
                                           parameter.

# Examples
```julia
opt = ADADelta()

opt = ADADelta(0.89)
```
"""
mutable struct ADADelta
  rho::Float64
  state::IdDict
  rho_dict::IdDict
end

ADADelta(ρ = 0.9; ρ_dict = IdDict()) = ADADelta(ρ, IdDict(), ρ_dict)

function apply!(o::ADADelta, x, Δ)
  ρ = get(o.rho_dict, x, o.rho)
  acc, Δacc = get!(o.state, x, (zero(x), zero(x)))
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= √Δacc/ (√acc + ϵ)
  @. Δacc = ρ * Δacc + (1 - ρ) * Δ^2
  return Δ
end

"""
    AMSGrad(η = 0.001, β::Tuple = (0.9, 0.999); η_dict = IdDict(), β_dict = IdDict())

The [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) version of the ADAM
optimiser. Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter momentum decay (`β_dict`): Same as the decay of momentum but
                                           specific for each parameter.

# Examples
```julia
opt = AMSGrad()

opt = AMSGrad(0.001, (0.89, 0.995))
```
"""
mutable struct AMSGrad
  eta::Float64
  beta::Tuple{Float64, Float64}
  state::IdDict
  eta_dict::IdDict
  beta_dict::IdDict
end

AMSGrad(η = 0.001, β = (0.9, 0.999); η_dict = IdDict(), β_dict = IdDict()) = AMSGrad(η, β, IdDict(), η_dict, β_dict)

function apply!(o::AMSGrad, x, Δ)
  η, β = get(o.eta_dict, x, o.eta), get(o.beta_dict, x, o.beta)
  mt, vt, v̂t = get!(o.state, x, (fill!(zero(x), ϵ), fill!(zero(x), ϵ), fill!(zero(x), ϵ)))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ ^ 2
  @. v̂t = max(v̂t, vt)
  @. Δ = η * mt / (√v̂t + ϵ)
end

"""
    NADAM(η = 0.001, β::Tuple = (0.9, 0.999); η_dict = IdDict(), β_dict = IdDict())

[NADAM](http://cs229.stanford.edu/proj2015/054_report.pdf) is a Nesterov variant of ADAM.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter momentum decay (`β_dict`): Same as the decay of momentum but
                                           specific for each parameter.

# Examples
```julia
opt = NADAM()

opt = NADAM(0.002, (0.89, 0.995))
```
"""
mutable struct NADAM
  eta::Float64
  beta::Tuple{Float64, Float64}
  state::IdDict
  eta_dict::IdDict
  beta_dict::IdDict
end

NADAM(η = 0.001, β = (0.9, 0.999); η_dict = IdDict(), β_dict = IdDict()) = NADAM(η, β, IdDict(), η_dict, β_dict)

function apply!(o::NADAM, x, Δ)
  η, β = get(o.eta_dict, x, o.eta), get(o.beta_dict, x, o.beta)
  mt, vt, (β1p, β2p) = get!(o.state, x, (zero(x), zero(x), o.beta))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ = (β[1] * mt / (1 - β[1] * β1p) + (1 - β[1]) * Δ / (1 - β1p)) / (√(vt * β[2] / (1 - β2p)) + ϵ) * η
  o.state[x] = (mt, vt, (β1p * β[1], β2p * β[2]))
  return Δ
end

"""
    ADAMW(η = 0.001, β::Tuple = (0.9, 0.999), decay = 0; η_dict = IdDict(), β_dict = IdDict(), decay_dict = IdDict())

[ADAMW](https://arxiv.org/abs/1711.05101) is a variant of ADAM fixing (as in repairing) its
weight decay regularization.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- `decay`: Decay applied to weights during optimisation.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter momentum decay (`β_dict`): Same as the decay of momentum but
                                           specific for each parameter.
- Per-parameter decay (`decay_dict`): Same as the decay but specific for each
                                      parameter.

# Examples
```julia
opt = ADAMW()

opt = ADAMW(0.001, (0.89, 0.995), 0.1)
```
"""
ADAMW(η = 0.001, β = (0.9, 0.999), decay = 0; η_dict = IdDict(), β_dict = IdDict(), decay_dict = IdDict()) =
  Optimiser(ADAM(η, β; η_dict = η_dict, β_dict = β_dict), WeightDecay(decay; decay_dict = decay_dict))

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
    InvDecay(γ = 0.001; γ_dict = IdDict())

Apply inverse time decay to an optimiser, so that the effective step size at
iteration `n` is `eta / (1 + γ * n)` where `eta` is the initial step size.
The wrapped optimiser's step size is not modified.

# Parameters
- Decay rate (`γ`): Multiplicative factor for the decay calculation.
- Per-parameter decay rate (`γ_dict`): Same as the decay rate but specific for
                                       each parameter.

# Examples
```julia
Optimiser(InvDecay(..), Opt(..))
```
"""
mutable struct InvDecay
  gamma::Float64
  state::IdDict
  gamma_dict::IdDict
end

InvDecay(γ = 0.001; γ_dict = IdDict()) = InvDecay(γ, IdDict(), γ_dict)

function apply!(o::InvDecay, x, Δ)
  γ = get(o.gamma_dict, x, o.gamma)
  n = get!(o.state, x, 1)
  Δ .*= 1 / (1 + γ * n)
  o.state[x] = n + 1
  return Δ
end

"""
    ExpDecay(η = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4;
    η_dict = IdDict(), decay_dict = IdDict(), step_dict = IdDict(), clip_dict = IdDict())

Discount the learning rate `η` by the factor `decay` every `decay_step` steps till
a minimum of `clip`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- `decay`: Factor by which the learning rate is discounted.
- `decay_step`: Schedule decay operations by setting the number of steps between
                two decay operations.
- `clip`: Minimum value of learning rate.
- Per-parameter learning rates (`η_dict`): Same as the learning rate but
                                           specific for each parameter.
- Per-parameter decay (`decay_dict`): Same as the decay but specific for each
                                      parameter.
- Per-parameter step (`step_dict`): Same as the step but specific for each
                                    parameter.
- Per-parameter clip (`clip_dict`): Same as the clip but specific for each
                                    parameter.

# Examples
To apply exponential decay to an optimiser:
```julia
Optimiser(ExpDecay(..), Opt(..))

opt = Optimiser(ExpDecay(), ADAM())
```
"""
mutable struct ExpDecay
  eta::Float64
  decay::Float64
  step::Int64
  clip::Float64
  current::IdDict
  eta_dict::IdDict
  decay_dict::IdDict
  step_dict::IdDict
  clip_dict::IdDict
end

ExpDecay(opt = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4;
η_dict = IdDict(), decay_dict = IdDict(), step_dict = IdDict(), clip_dict = IdDict()) = 
  ExpDecay(opt, decay, decay_step, clip, IdDict(), η_dict, decay_dict, step_dict, clip_dict)

function apply!(o::ExpDecay, x, Δ)
  η, s, decay, clip = get(o.eta_dict, x, o.eta), get(o.step_dict, x, o.step), get(o.decay_dict, x, o.decay), get(o.clip_dict, x, o.clip)
  n = o.current[x] = get(o.current, x, 0) + 1
  if o.current[x]%s == 0 && count(x -> x%s == 0, values(o.current)) == 1
    η = max(η * decay, clip)
    o.eta = η
  end
  @. Δ *= η
end

"""
    WeightDecay(wd = 0; decay_dict = IdDict())

Decay weights by `wd`.

# Parameters
- Weight decay (`wd`)
- Per-parameter weigth decay (`decay_dict`): Same as weight decay but specific
                                             for each parameter (weight).
"""
mutable struct WeightDecay
  wd::Real
  decay_dict::IdDict
end

WeightDecay(decay = 0; decay_dict = IdDict()) = WeightDecay(decay, decay_dict)

function apply!(o::WeightDecay, x, Δ)
  wd = get(o.decay_dict, x, o.wd)
  @. Δ += wd * x
end

"""
    ClipValue(thresh; thresh_dict = IdDict())

Clip gradients when their absolute value exceeds `thresh`. Thresholds can be
specified for individual parameters through thresh_dict.
"""
mutable struct ClipValue{T}
  thresh::T
  thresh_dict::IdDict
end

ClipValue(thresh; thresh_dict = IdDict()) = ClipValue(thresh, thresh_dict)

function apply!(o::ClipValue, x, Δ)
  thresh = get(o.thresh_dict, x, o.thresh)
  clamp!(Δ, -thresh, thresh)
end

"""
    ClipNorm(thresh; thresh_dict = IdDict())

Clip gradients when their L2 norm exceeds `thresh`. Thresholds can be specified
for individual parameters through thresh_dict.
"""
mutable struct ClipNorm{T}
  thresh::T
  thresh_dict::IdDict
end

ClipNorm(thresh; thresh_dict = IdDict()) = ClipNorm(thresh, thresh_dict)

function apply!(o::ClipNorm, x, Δ)
  thresh = get(o.thresh_dict, x, o.thresh)
  Δnrm = norm(Δ)
  if Δnrm > o.thresh
      rmul!(Δ, thresh / Δnrm)
  end
  return Δ
end