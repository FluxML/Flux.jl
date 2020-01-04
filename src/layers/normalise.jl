istraining() = false

@adjoint istraining() = true, _ -> nothing

_dropout_shape(s, ::Colon) = size(s)
_dropout_shape(s, dims) = tuple((i ∉ dims ? 1 : si for (i, si) ∈ enumerate(size(s)))...)

_dropout_kernel(y::T, p, q) where {T} = y > p ? T(1 / q) : T(0)

dropout(x, p; dims = :) = x

@adjoint function dropout(x, p; dims = :)
  y = rand!(similar(x, _dropout_shape(x, dims)))
  y .= _dropout_kernel.(y, p, 1 - p)
  return x .* y, Δ -> (Δ .* y, nothing)
end

"""
    Dropout(p, dims = :)

A Dropout layer. For each input, either sets that input to `0` (with probability
`p`) or scales it by `1/(1-p)`. The `dims` argument is to specified the unbroadcasted
 dimensions, i.e. `dims=1` does dropout along columns and `dims=2` along rows. This is
 used as a regularisation, i.e. it reduces overfitting during training. see also [`dropout`](@ref).
"""
mutable struct Dropout{F,D}
  p::F
  dims::D
end

function Dropout(p; dims = :)
  @assert 0 ≤ p ≤ 1
  Dropout{typeof(p),typeof(dims)}(p, dims)
end

(a::Dropout)(x) = dropout(x, a.p; dims = a.dims)

function Base.show(io::IO, d::Dropout)
  print(io, "Dropout(", d.p)
  d.dims != (:) && print(io, ", dims = $(repr(d.dims))")
  print(io, ")")
end

"""
    AlphaDropout(p)
A dropout layer. It is used in Self-Normalizing Neural Networks.
(https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf)
The AlphaDropout layer ensures that mean and variance of activations remains the same as before.
"""
mutable struct AlphaDropout{F}
  p::F
  function AlphaDropout(p)
    @assert 0 ≤ p ≤ 1
    new{typeof(p)}(p)
  end
end

function (a::AlphaDropout)(x)
  istraining() || return x
  λ = eltype(x)(1.0507009873554804934193349852946)
  α = eltype(x)(1.6732632423543772848170429916717)
  α1 = eltype(x)(-λ*α)
  noise = randn(eltype(x), size(x))
  x = @. x*(noise > (1 - a.p)) + α1 * (noise < (1 - a.p))
  A = (a.p + a.p * (1 - a.p) * α1 ^ 2)^0.5
  B = -A * α1 * (1 - a.p)
  x = @. A * x + B
  return x
end

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

@functor LayerNorm

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
end

BatchNorm(chs::Integer, λ = identity;
          initβ = (i) -> zeros(Float32, i), initγ = (i) -> ones(Float32, i), ϵ = 1f-5, momentum = 0.1f0) =
  BatchNorm(λ, initβ(chs), initγ(chs),
            zeros(chs), ones(chs), ϵ, momentum)

trainable(bn::BatchNorm) = (bn.β, bn.γ)

function (BN::BatchNorm)(x)
  size(x, ndims(x)-1) == length(BN.β) ||
    error("BatchNorm expected $(length(BN.β)) channels, got $(size(x, ndims(x)-1))")
  dims = length(size(x))
  channels = size(x, dims-1)
  affine_shape = ntuple(i->i == ndims(x) - 1 ? size(x, i) : 1, ndims(x))
  m = div(prod(size(x)), channels)
  γ = reshape(BN.γ, affine_shape...)
  β = reshape(BN.β, affine_shape...)
  if !istraining()
    μ = reshape(BN.μ, affine_shape...)
    σ² = reshape(BN.σ², affine_shape...)
    ϵ = BN.ϵ
  else
    T = eltype(x)
    axes = [1:dims-2; dims] # axes to reduce along (all but channels axis)
    μ = mean(x, dims = axes)
    σ² = sum((x .- μ) .^ 2, dims = axes) ./ m
    ϵ = convert(T, BN.ϵ)
    # update moving mean/std
    mtm = BN.momentum
    S = eltype(BN.μ)
    BN.μ  = (1 - mtm) .* BN.μ .+ mtm .* S.(reshape(μ, :))
    BN.σ² = (1 - mtm) .* BN.σ² .+ (mtm * m / (m - 1)) .* S.(reshape(σ², :))
  end

  let λ = BN.λ
    x̂ = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    λ.(γ .* x̂ .+ β)
  end
end

@functor BatchNorm

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
expand_inst = (x, as) -> reshape(repeat(x, outer=[1, as[length(as)]]), as...)

mutable struct InstanceNorm{F,V,W,N}
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::W  # moving mean
  σ²::W  # moving std
  ϵ::N
  momentum::N
end

InstanceNorm(chs::Integer, λ = identity;
          initβ = (i) -> zeros(Float32, i), initγ = (i) -> ones(Float32, i), ϵ = 1f-5, momentum = 0.1f0) =
  InstanceNorm(λ, initβ(chs), initγ(chs),
            zeros(chs), ones(chs), ϵ, momentum)

trainable(in::InstanceNorm) = (in.β, in.γ)

function (in::InstanceNorm)(x)
  size(x, ndims(x)-1) == length(in.β) ||
    error("InstanceNorm expected $(length(in.β)) channels, got $(size(x, ndims(x)-1))")
  ndims(x) > 2 ||
    error("InstanceNorm requires at least 3 dimensions. With 2 dimensions an array of zeros would be returned")
  # these are repeated later on depending on the batch size
  dims = length(size(x))
  c = size(x, dims-1)
  bs = size(x, dims)
  affine_shape = ntuple(i->i == ndims(x) - 1 || i == ndims(x) ? size(x, i) : 1, ndims(x))
  m = div(prod(size(x)), c*bs)
  γ, β = expand_inst(in.γ, affine_shape), expand_inst(in.β, affine_shape)

  if !istraining()
    μ = expand_inst(in.μ, affine_shape)
    σ² = expand_inst(in.σ², affine_shape)
    ϵ = in.ϵ
  else
    T = eltype(x)

    ϵ = convert(T, in.ϵ)
    axes = 1:dims-2 # axes to reduce along (all but channels and batch size axes)
    μ = mean(x, dims = axes)
    σ² = mean((x .- μ) .^ 2, dims = axes)
    S = eltype(in.μ)
    # update moving mean/std
    mtm = in.momentum
    in.μ = dropdims(mean(repeat((1 - mtm) .* in.μ, outer=[1, bs]) .+ mtm .* S.(reshape(μ, (c, bs))), dims = 2), dims=2)
    in.σ² = dropdims(mean((repeat((1 - mtm) .* in.σ², outer=[1, bs]) .+ (mtm * m / (m - 1)) .* S.(reshape(σ², (c, bs)))), dims = 2), dims=2)
  end

  let λ = in.λ
    x̂ = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    λ.(γ .* x̂ .+ β)
  end
end

@functor InstanceNorm

function Base.show(io::IO, l::InstanceNorm)
  print(io, "InstanceNorm($(join(size(l.β), ", "))")
  (l.λ == identity) || print(io, ", λ = $(l.λ)")
  print(io, ")")
end

"""
Group Normalization.
This layer can outperform Batch-Normalization and Instance-Normalization.

	GroupNorm(chs::Integer, G::Integer, λ = identity;
	          initβ = (i) -> zeros(Float32, i), initγ = (i) -> ones(Float32, i),
	          ϵ = 1f-5, momentum = 0.1f0)

``chs`` is the number of channels, the channel dimension of your input.
For an array of N dimensions, the (N-1)th index is the channel dimension.

``G`` is the number of groups along which the statistics would be computed.
The number of channels must be an integer multiple of the number of groups.

Example:
```
m = Chain(Conv((3,3), 1=>32, leakyrelu;pad = 1),
          GroupNorm(32,16)) # 32 channels, 16 groups (G = 16), thus 2 channels per group used
```

Link : https://arxiv.org/pdf/1803.08494.pdf
"""
mutable struct GroupNorm{F,V,W,N,T}
  G::T # number of groups
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::W  # moving mean
  σ²::W  # moving std
  ϵ::N
  momentum::N
end

GroupNorm(chs::Integer, G::Integer, λ = identity;
          initβ = (i) -> zeros(Float32, i), initγ = (i) -> ones(Float32, i), ϵ = 1f-5, momentum = 0.1f0) =
  GroupNorm(G, λ, initβ(chs), initγ(chs),
            zeros(G,1), ones(G,1), ϵ, momentum)

trainable(gn::GroupNorm) = (gn.β, gn.γ)

function(gn::GroupNorm)(x)
  size(x,ndims(x)-1) == length(gn.β) || error("Group Norm expected $(length(gn.β)) channels, but got $(size(x,ndims(x)-1)) channels")
  ndims(x) > 2 || error("Need to pass at least 3 channels for Group Norm to work")
  (size(x,ndims(x) -1))%gn.G == 0 || error("The number of groups ($(gn.G)) must divide the number of channels ($(size(x,ndims(x) -1)))")

  dims = length(size(x))
  groups = gn.G
  channels = size(x, dims-1)
  batches = size(x,dims)
  channels_per_group = div(channels,groups)
  affine_shape = ntuple(i->i == ndims(x) - 1 ? size(x, i) : 1, ndims(x))

  # Output reshaped to (W,H...,C/G,G,N)
  μ_affine_shape = ntuple(i->i == ndims(x) ? groups : 1, ndims(x) + 1)

  m = prod(size(x)[1:end-2]) * channels_per_group
  γ = reshape(gn.γ, affine_shape...)
  β = reshape(gn.β, affine_shape...)

  y = reshape(x,((size(x))[1:end-2]...,channels_per_group,groups,batches))
  if !istraining()
    og_shape = size(x)
    μ = reshape(gn.μ, μ_affine_shape...) # Shape : (1,1,...C/G,G,1)
    σ² = reshape(gn.σ², μ_affine_shape...) # Shape : (1,1,...C/G,G,1)
    ϵ = gn.ϵ
  else
    T = eltype(x)
    og_shape = size(x)
    axes = [(1:ndims(y)-2)...] # axes to reduce along (all but channels axis)
    μ = mean(y, dims = axes)
    σ² = mean((y .- μ) .^ 2, dims = axes)

    ϵ = convert(T, gn.ϵ)
    # update moving mean/std
    mtm = gn.momentum
    S = eltype(gn.μ)
    gn.μ = mean((1 - mtm) .* gn.μ .+ mtm .* S.(reshape(μ, (groups,batches))),dims=2)
    gn.σ² = mean((1 - mtm) .* gn.σ² .+ (mtm * m / (m - 1)) .* S.(reshape(σ², (groups,batches))),dims=2)
  end

  let λ = gn.λ
    x̂ = (y .- μ) ./ sqrt.(σ² .+ ϵ)

    # Reshape x̂
    x̂ = reshape(x̂,og_shape)
    λ.(γ .* x̂ .+ β)
  end
end

@functor GroupNorm

function Base.show(io::IO, l::GroupNorm)
  print(io, "GroupNorm($(join(size(l.β), ", "))")
  (l.λ == identity) || print(io, ", λ = $(l.λ)")
  print(io, ")")
end
