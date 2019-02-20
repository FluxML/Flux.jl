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

_dropout_kernel(y::T, p, q) where {T} = y > p ? T(1 / q) : T(0)

function (a::Dropout)(x)
  a.active || return x
  y = similar(x)
  rand!(y)
  y .= _dropout_kernel.(y, a.p, 1 - a.p)
  return x .* y
end

_testmode!(a::Dropout, test) = (a.active = !test)

"""
    LayerNorm(h::Integer)

A [normalisation layer](https://arxiv.org/pdf/1607.06450.pdf) designed to be
used with recurrent hidden states of size `h`. Normalises the mean/stddev of
each input before applying a per-neuron gain/bias.
"""
struct LayerNorm{T}
  diag::Diagonal{T}
end

LayerNorm(h::Integer) =
  LayerNorm(Diagonal(h))

@treelike LayerNorm

(a::LayerNorm)(x) = a.diag(normalise(x))

function Base.show(io::IO, l::LayerNorm)
  print(io, "LayerNorm(", length(l.diag.α), ")")
end

"""
    BatchNorm(channels::Integer, σ = identity;
              initβ = zeros, initγ = ones,
              ϵ = 1e-8, momentum = .1)

Batch Normalization layer. The `channels` input should be the size of the
channel dimension in your data (see below).

Given an array with `N` dimensions, call the `N-1`th the channel dimension. (For
a batch of feature vectors this is just the data dimension, for `WHCN` images
it's the usual channel dimension.)

`BatchNorm` computes the mean and variance for each each `W×H×1×N` slice and
shifts them to have a new mean and variance (corresponding to the learnable,
per-channel `bias` and `scale` parameters).

See [Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf).

Example:
```julia
m = Chain(
  Dense(28^2, 64),
  BatchNorm(64, relu),
  Dense(64, 10),
  BatchNorm(10),
  softmax)
```
"""
mutable struct BatchNorm{F,V,W,N}
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::W  # moving mean
  σ²::W  # moving std
  ϵ::N
  momentum::N
  active::Bool
end

BatchNorm(chs::Integer, λ = identity;
          initβ = (i) -> zeros(Float32, i), initγ = (i) -> ones(Float32, i), ϵ = 1f-5, momentum = 0.1f0) =
  BatchNorm(λ, param(initβ(chs)), param(initγ(chs)),
            zeros(chs), ones(chs), ϵ, momentum, true)

function (BN::BatchNorm)(x)
  size(x, ndims(x)-1) == length(BN.β) ||
    error("BatchNorm expected $(length(BN.β)) channels, got $(size(x, ndims(x)-1))")
  dims = length(size(x))
  channels = size(x, dims-1)
  affine_shape = ones(Int, dims)
  affine_shape[end-1] = channels
  m = prod(size(x)[1:end-2]) * size(x)[end]
  γ = reshape(BN.γ, affine_shape...)
  β = reshape(BN.β, affine_shape...)
  if !BN.active
    μ = reshape(BN.μ, affine_shape...)
    σ² = reshape(BN.σ², affine_shape...)
    ϵ = BN.ϵ
  else
    T = eltype(x)
    axes = [1:dims-2; dims] # axes to reduce along (all but channels axis)
    μ = mean(x, dims = axes)
    σ² = sum((x .- μ) .^ 2, dims = axes) ./ m
    ϵ = data(convert(T, BN.ϵ))
    # update moving mean/std
    mtm = data(convert(T, BN.momentum))
    BN.μ = (1 - mtm) .* BN.μ .+ mtm .* reshape(data(μ), :)
    BN.σ² = (1 - mtm) .* BN.σ² .+ (mtm * m / (m - 1)) .* reshape(data(σ²), :)
  end

  let λ = BN.λ
    x̂ = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    λ.(γ .* x̂ .+ β)
  end
end

children(BN::BatchNorm) =
  (BN.λ, BN.β, BN.γ, BN.μ, BN.σ², BN.ϵ, BN.momentum, BN.active)

mapchildren(f, BN::BatchNorm) =  # e.g. mapchildren(cu, BN)
  BatchNorm(BN.λ, f(BN.β), f(BN.γ), f(BN.μ), f(BN.σ²), BN.ϵ, BN.momentum, BN.active)

_testmode!(BN::BatchNorm, test) = (BN.active = !test)

function Base.show(io::IO, l::BatchNorm)
  print(io, "BatchNorm($(join(size(l.β), ", "))")
  (l.λ == identity) || print(io, ", λ = $(l.λ)")
  print(io, ")")
end


"""
    InstanceNorm(channels::Integer, σ = identity;
                 initβ = zeros, initγ = ones,
                 ϵ = 1e-8, momentum = .1)

Instance Normalization layer. The `channels` input should be the size of the
channel dimension in your data (see below).

Given an array with `N` dimensions, call the `N-1`th the channel dimension. (For
a batch of feature vectors this is just the data dimension, for `WHCN` images
it's the usual channel dimension.)

`InstanceNorm` computes the mean and variance for each each `W×H×1×1` slice and
shifts them to have a new mean and variance (corresponding to the learnable,
per-channel `bias` and `scale` parameters).

See [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).

Example:
```julia
m = Chain(
  Dense(28^2, 64),
  InstanceNorm(64, relu),
  Dense(64, 10),
  InstanceNorm(10),
  softmax)
```
"""
mutable struct InstanceNorm{F,V,W,N}
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::W  # moving mean
  σ²::W  # moving std
  ϵ::N
  momentum::N
  active::Bool
end

InstanceNorm(chs::Integer, λ = identity;
          initβ = (i) -> zeros(Float32, i), initγ = (i) -> ones(Float32, i), ϵ = 1f-5, momentum = 0.1f0) =
  InstanceNorm(λ, param(initβ(chs)), param(initγ(chs)),
            zeros(chs), ones(chs), ϵ, momentum, true)

function (IN::InstanceNorm)(x)
  size(x, ndims(x)-1) == length(IN.β) ||
    error("InstanceNorm expected $(length(IN.β)) channels, got $(size(x, ndims(x)-1))")
  ndims(x) > 2 ||
    error("InstanceNorm requires at least 3 dimensions. With 2 dimensions an array of zeros would be returned")
  # these are repeated later on depending on the batch size
  γ, β = IN.γ, IN.β
  dims = length(size(x))
  c = size(x, dims-1)
  bs = size(x, dims)
  affine_shape = ones(Int, dims)
  affine_shape[end-1] = c
  affine_shape[end] = bs
  m = prod(size(x)[1:end-2])

  if !IN.active
    μ = reshape(repeat(IN.μ, outer=[bs]), affine_shape...)
    σ² = reshape(repeat(IN.σ², outer=[bs]), affine_shape...)
  else
    T = eltype(x)

    ϵ = data(convert(T, IN.ϵ))
    axes = 1:dims-2 # axes to reduce along (all but channels and batch size axes)
    μ = mean(x, dims = axes)
    σ² = mean((x .- μ) .^ 2, dims = axes)

    # update moving mean/std
    mtm = data(convert(T, IN.momentum))
    IN.μ = reshape(mean((1 - mtm) .* repeat(IN.μ, outer=[1, bs]) .+ mtm .* reshape(data(μ), (c, bs)), dims = 2), :)
    IN.σ² = reshape(mean(((1 - mtm) .* repeat(IN.σ², outer=[1, bs]) .+ mtm .* reshape(data(σ²), (c, bs)) .* (m / (m - 1))), dims = 2), :)
  end

  let λ = IN.λ
    temp = reshape(repeat(γ, outer=[bs]), affine_shape...) .* ((x .- μ) ./ sqrt.(σ² .+ IN.ϵ))
    # This is intentionally not fused because of an extreme slowdown doing so
    λ.(temp .+ reshape(repeat(β, outer=[bs]), affine_shape...))
  end
end

children(IN::InstanceNorm) =
  (IN.λ, IN.β, IN.γ, IN.μ, IN.σ², IN.ϵ, IN.momentum, IN.active)

mapchildren(f, IN::InstanceNorm) =  # e.g. mapchildren(cu, IN)
  InstanceNorm(IN.λ, f(IN.β), f(IN.γ), f(IN.μ), f(IN.σ²), IN.ϵ, IN.momentum, IN.active)

_testmode!(IN::InstanceNorm, test) = (IN.active = !test)

function Base.show(io::IO, l::InstanceNorm)
  print(io, "InstanceNorm($(join(size(l.β), ", "))")
  (l.λ == identity) || print(io, ", λ = $(l.λ)")
  print(io, ")")
end
