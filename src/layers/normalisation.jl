"""
    testmode!(m)
    testmode!(m, false)

Put layers like [`Dropout`](@ref) and [`BatchNorm`](@ref) into testing mode
(or back to training mode with `false`).
"""
function testmode!(m, val::Bool=true)
  prefor(x -> _testmode!(x, val), m)
  return m
end

_testmode!(m, test) = nothing

"""
    Dropout(p)

A Dropout layer. For each input, either sets that input to `0` (with probability
`p`) or scales it by `1/(1-p)`. This is used as a regularisation, i.e. it
reduces overfitting during training.

Does nothing to the input once in [`testmode!`](@ref).
"""
mutable struct Dropout{F}
  p::F
  active::Bool
end

function Dropout(p)
  @assert 0 ≤ p ≤ 1
  Dropout{typeof(p)}(p, true)
end

function (a::Dropout)(x)
  a.active || return x
  y = similar(x)
  rand!(y)
  q = 1 - a.p
  @inbounds for i=1:length(y)
    y[i] = y[i] > a.p ? 1 / q : 0
  end
  return y .* x
end

_testmode!(a::Dropout, test) = (a.active = !test)

"""
    BatchNorm(dims...; λ = identity,
              initβ = zeros, initγ = ones, ϵ = 1e-8, momentum = .1)

Batch Normalization Layer for [`Dense`](@ref) layer.

See [Batch Normalization: Accelerating Deep Network Training by Reducing
     Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

In the example of MNIST,
in order to normalize the input of other layer,
put the `BatchNorm` layer before activation function.

```julia
julia> m = Chain(
  Dense(28^2, 64),
  BatchNorm(64, λ = relu),
  Dense(64, 10),
  BatchNorm(10),
  softmax)
Chain(Dense(784, 64), BatchNorm(64, λ = NNlib.relu), Dense(64, 10), BatchNorm(10), NNlib.softmax)

julia> opt = SGD(params(m), 10)  # a crazy learning rate
```
"""
mutable struct BatchNorm{F,V,N}
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ     # moving mean
  σ     # moving std
  ϵ::N
  momentum::N
  active::Bool
end

BatchNorm(dims::Integer...; λ = identity,
          initβ = zeros, initγ = ones, ϵ = 1e-8, momentum = .1) =
  BatchNorm(λ, param(initβ(dims)), param(initγ(dims)), 0., 1., ϵ, momentum, true)

function (BN::BatchNorm)(x)
  if !BN.active
    μ = BN.μ
    σ = BN.σ
  else
    T = eltype(x)

    ϵ = T(BN.ϵ)
    m = size(x, 2)  # batch size
    μ = sum(x, 2) ./ m
    σ = sqrt.(sum((x .- μ).^2, 2) ./ m .+ ϵ)

    # update moving mean/std
    mtm = T(BN.momentum)
    BN.μ = (1 - mtm) .* BN.μ .+ mtm .* μ.data
    BN.σ = (1 - mtm) .* BN.σ .+ mtm .* σ.data .* m ./ (m - 1)
  end

  BN.λ.(BN.γ .* ((x .- μ) ./ σ) .+ BN.β)
end

children(BN::BatchNorm) =
  (BN.λ, BN.β, BN.γ, BN.μ, BN.σ, BN.momentum, BN.ϵ, BN.active)

mapchildren(f, BN::BatchNorm) =  # e.g. mapchildren(cu, BN)
  BatchNorm(BN.λ, f(BN.β), f(BN.γ), BN.μ, BN.σ, BN.momentum, BN.ϵ, BN.active)

_testmode!(BN::BatchNorm, test) = (BN.active = !test)

function Base.show(io::IO, l::BatchNorm)
  print(io, "BatchNorm($(join(size(l.β), ", "))")
  (l.λ == identity) || print(io, ", λ = $(l.λ)")
  print(io, ")")
end
