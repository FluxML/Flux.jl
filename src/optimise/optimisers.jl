using Flux
using Base: @get!

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

function update!(o::Descent, x, Δ)
  Δ .*= o.eta
end

"""
    Momentum(params, η = 0.01; ρ = 0.9, decay = 0)

Gradient descent with learning rate `η` and momentum `ρ`.
"""
mutable struct Momentum
  eta::Float64
  rho::Float64
  velocity::ObjectIdDict
end

Momentum(η, ρ = 0.9) = Momentum(η, ρ, ObjectIdDict())

function update!(o::Momentum, x, Δ)
  η, ρ = o.eta, o.rho
  v = @get!(o.velocity, x, zero(x))::typeof(x)
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
  velocity::ObjectIdDict
end

Nesterov(η, ρ = 0.9) = Nesterov(η, ρ, ObjectIdDict())

function update!(o::Nesterov, x, Δ)
  η, ρ = o.eta, o.rho
  v = @get!(o.velocity, x, zero(x))::typeof(x)
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
  acc::ObjectIdDict
end

RMSProp(η = 0.001, ρ = 0.9) = RMSProp(η, ρ, ObjectIdDict())

function update!(o::RMSProp, x, Δ)
  η, ρ = o.eta, o.rho
  acc = @get!(o.acc, x, zero(x))::typeof(x)
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= η / (√acc + ϵ)
end

"""
    ADAM(params, η = 0.001; β1 = 0.9, β2 = 0.999, ϵ = 1e-08, decay = 0)

[ADAM](https://arxiv.org/abs/1412.6980v8) optimiser.
"""
mutable struct ADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::ObjectIdDict
end

ADAM(η = 0.001, β = (0.9, 0.999)) = ADAM(η, β, ObjectIdDict())

function update!(o::ADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp = @get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  o.state[x] = (mt, vt, βp .* β)
end

# """
#     AdaMax(params, η = 0.001; β1 = 0.9, β2 = 0.999, ϵ = 1e-08, decay = 0)
#
# [AdaMax](https://arxiv.org/abs/1412.6980v9) optimiser. Variant of ADAM based on
# the ∞-norm.
# """

# """
#     ADAGrad(params, η = 0.01; ϵ = 1e-8, decay = 0)
#
# [ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimiser.
# Parameters don't need tuning.
# """

# """
#     ADADelta(params; ρ = 0.9, ϵ = 1e-8, decay = 0)
#
# [ADADelta](http://arxiv.org/abs/1212.5701) optimiser. Parameters don't need
# tuning.
# """

# """
#     AMSGrad(params; η = 0.001, β1 = 0.9, β2 = 0.999, ϵ = 1e-08, decay = 0)
#
# [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) optimiser. Parameters don't need
# tuning.
# """

# struct Optimiser
#   os::Vector{Any}
# end

# TODO: decay
