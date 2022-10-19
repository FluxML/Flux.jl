istraining() = false

ChainRulesCore.rrule(::typeof(istraining)) = true, _ -> (NoTangent(),)

_isactive(m) = isnothing(m.active) ? istraining() : m.active

_dropout_shape(s, ::Colon) = size(s)
_dropout_shape(s, dims) = tuple((i ∉ dims ? 1 : si for (i, si) in enumerate(size(s)))...)

_dropout_kernel(y::T, p, q) where {T} = y > p ? T(1 / q) : T(0)

"""
    dropout([rng = rng_from_array(x)], x, p; dims=:, active=true)

The dropout function. If `active` is `true`,
for each input, either sets that input to `0` (with probability
`p`) or scales it by `1 / (1 - p)`. `dims` specifies the unbroadcasted dimensions,
e.g. `dims=1` applies dropout along columns and `dims=2` along rows.
If `active` is `false`, it just returns the input `x`.

Specify `rng` for custom RNGs instead of the default RNG.
Note that custom RNGs are only supported on the CPU.

Warning: when using this function, you have to manually manage the activation
state. Usually in fact, dropout is used while training
but is deactivated in the inference phase. This can be
automatically managed using the [`Dropout`](@ref) layer instead of the
`dropout` function.

The [`Dropout`](@ref) layer is what you should use in most scenarios.
"""
function dropout(rng, x, p; dims = :, active::Bool = true)
    active || return x
    y = dropout_mask(rng, x, p, dims = dims)
    return x .* y
end
dropout(x, p; kwargs...) = dropout(rng_from_array(x), x, p; kwargs...)

dropout_mask(rng::CUDA.RNG, x::CuArray, p; kwargs...) = _dropout_mask(rng, x, p; kwargs...)
function dropout_mask(rng, x::CuArray, p; kwargs...)
    throw(ArgumentError("x isa CuArray, but rng isa $(typeof(rng)). dropout_mask only support CUDA.RNG for CuArrays."))
end
dropout_mask(rng, x, p; kwargs...) = _dropout_mask(rng, x, p; kwargs...)
function _dropout_mask(rng, x, p; dims = :)
    realfptype = float(real(eltype(x)))
    y = rand!(rng, similar(x, realfptype, _dropout_shape(x, dims)))
    y .= _dropout_kernel.(y, p, 1 - p)
    return y
end

# TODO move this to NNlib
ChainRulesCore.@non_differentiable dropout_mask(::Any, ::Any, ::Any)

"""
    Dropout(p; dims=:, rng = default_rng_value())

Dropout layer.

While training, for each input, this layer either sets that input to `0` (with probability
`p`) or scales it by `1 / (1 - p)`. To apply dropout along certain dimension(s), specify the
`dims` keyword. e.g. `Dropout(p; dims = 3)` will randomly zero out entire channels on WHCN input
(also called 2D dropout). This is used as a regularisation, i.e. it reduces overfitting during
training.

In the forward pass, this layer applies the [`Flux.dropout`](@ref) function. See that for more
details.

Specify `rng` to use a custom RNG instead of the default.
Custom RNGs are only supported on the CPU.

Does nothing to the input once [`Flux.testmode!`](@ref) is `true`.

# Examples

```jldoctest
julia> m = Chain(Dense(1 => 1), Dropout(1));

julia> Flux.trainmode!(m);

julia> y = m([1]);

julia> y == [0]
true

julia> m = Chain(Dense(1000 => 1000), Dropout(0.5));

julia> Flux.trainmode!(m);

julia> y = m(ones(1000));

julia> isapprox(count(==(0), y) / length(y), 0.5, atol = 0.1)
true
```
"""
mutable struct Dropout{F, D, R <: AbstractRNG}
    p::F
    dims::D
    active::Union{Bool, Nothing}
    rng::R
end
Dropout(p, dims, active) = Dropout(p, dims, active, default_rng_value())

function Dropout(p; dims = :, rng = default_rng_value())
    @assert 0 ≤ p ≤ 1
    return Dropout(p, dims, nothing, rng)
end

@functor Dropout
trainable(a::Dropout) = (;)

function (a::Dropout)(x)
    _isactive(a) || return x
    return dropout(a.rng, x, a.p; dims = a.dims, active = true)
end

function testmode!(m::Dropout, mode = true)
    return (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
end

function Base.show(io::IO, d::Dropout)
    print(io, "Dropout(", d.p)
    d.dims != (:) && print(io, ", dims = $(repr(d.dims))")
    return print(io, ")")
end

"""
    AlphaDropout(p; rng = default_rng_value())

A dropout layer. Used in
[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515).
The AlphaDropout layer ensures that mean and variance of activations
remain the same as before.

Does nothing to the input once [`testmode!`](@ref) is true.

# Examples

```jldoctest
julia> using Statistics

julia> x = randn(1000, 1);

julia> m = Chain(Dense(1000 => 1000, selu), AlphaDropout(0.2));

julia> Flux.trainmode!(m);

julia> y = m(x);

julia> isapprox(std(x), std(y), atol = 0.2)
true
```
"""
mutable struct AlphaDropout{F, R <: AbstractRNG}
    p::F
    active::Union{Bool, Nothing}
    rng::R
    function AlphaDropout(p, active, rng)
        @assert 0 ≤ p ≤ 1
        return new{typeof(p), typeof(rng)}(p, active, rng)
    end
end
AlphaDropout(p, active) = AlphaDropout(p, active, default_rng_value())
AlphaDropout(p; rng = default_rng_value()) = AlphaDropout(p, nothing, rng)

@functor AlphaDropout
trainable(a::AlphaDropout) = (;)

function (a::AlphaDropout)(x::AbstractArray{T}) where {T}
    _isactive(a) || return x
    p = a.p
    iszero(p) && return x
    isone(p) && return sign.(x) .* T(0)

    α′ = T(-1.7580993408473766) # selu(-Inf) == -λα
    A = T(inv(sqrt((1 - p) * (1 + p * α′^2))))
    B = T(-A * α′ * p)

    noise = rand!(a.rng, similar(x))
    return A .* ifelse.(noise .> p, x, α′) .+ B
end

function testmode!(m::AlphaDropout, mode = true)
    return (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
end

"""
    LayerNorm(size..., λ=identity; affine=true, ϵ=1fe-5)

A [normalisation layer](https://arxiv.org/abs/1607.06450) designed to be
used with recurrent hidden states.
The argument `size` should be an integer or a tuple of integers.
In the forward pass, the layer normalises the mean and standard
deviation of the input, then applies the elementwise activation `λ`.
The input is normalised along the first `length(size)` dimensions
for tuple `size`, and along the first dimension for integer `size`.
The input is expected to have first dimensions' size equal to `size`.

If `affine=true`, it also applies a learnable shift and rescaling
using the [`Scale`](@ref) layer.

See also [`BatchNorm`](@ref), [`InstanceNorm`](@ref), [`GroupNorm`](@ref), and [`normalise`](@ref).

# Examples

```jldoctest
julia> using Statistics

julia> xs = rand(3, 3, 3, 2);  # a batch of 2 images, each having 3 channels

julia> m = LayerNorm(3);

julia> y = m(xs);

julia> isapprox(std(y, dims = 1:3), ones(1, 1, 1, 2), atol = 0.1) &&
           std(y, dims = 1:3) != std(xs, dims = 1:3)
true
```
"""
struct LayerNorm{F, D, T, N}
    λ::F
    diag::D
    ϵ::T
    size::NTuple{N, Int}
    affine::Bool
end

function LayerNorm(size::Tuple{Vararg{Int}}, λ = identity; affine::Bool = true,
                   ϵ::Real = 1.0f-5)
    diag = affine ? Scale(size..., λ) : λ != identity ? Base.Fix1(broadcast, λ) : identity
    return LayerNorm(λ, diag, ϵ, size, affine)
end
LayerNorm(size::Integer...; kw...) = LayerNorm(Int.(size); kw...)
LayerNorm(size_act...; kw...) = LayerNorm(Int.(size_act[1:(end - 1)]), size_act[end]; kw...)

@functor LayerNorm

(a::LayerNorm)(x) = a.diag(normalise(x, dims = 1:length(a.size), ϵ = a.ϵ))

function Base.show(io::IO, l::LayerNorm)
    print(io, "LayerNorm(", join(l.size, ", "))
    l.λ === identity || print(io, ", ", l.λ)
    hasaffine(l) || print(io, ", affine=false")
    return print(io, ")")
end

# For InstanceNorm, GroupNorm, and BatchNorm.
# Compute the statistics on the slices specified by reduce_dims.
# reduce_dims=[1,...,N-2,N] for BatchNorm
# reduce_dims=[1,...,N-2] for InstanceNorm and GroupNorm
function _norm_layer_forward(l, x::AbstractArray{T, N}; reduce_dims,
                             affine_shape) where {T, N}
    if !_isactive(l) && l.track_stats # testmode with tracked stats
        stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
        μ = reshape(l.μ, stats_shape)
        σ² = reshape(l.σ², stats_shape)
    else # trainmode or testmode without tracked stats
        μ = mean(x; dims = reduce_dims)
        σ² = var(x; mean = μ, dims = reduce_dims, corrected = false)
        if l.track_stats
            _track_stats!(l, x, μ, σ², reduce_dims) # update moving mean/std
        end
    end

    o = _norm_layer_forward(x, μ, σ², l.ϵ)
    hasaffine(l) || return l.λ.(o)

    γ = reshape(l.γ, affine_shape)
    β = reshape(l.β, affine_shape)
    return l.λ.(γ .* o .+ β)
end

@inline _norm_layer_forward(x, μ, σ², ϵ) = (x .- μ) ./ sqrt.(σ² .+ ϵ)

function _track_stats!(bn, x::AbstractArray{T, N}, μ, σ², reduce_dims) where {T, N}
    V = eltype(bn.σ²)
    mtm = bn.momentum
    res_mtm = one(V) - mtm
    m = prod(size(x, i) for i in reduce_dims)

    μnew = vec(N ∈ reduce_dims ? μ : mean(μ, dims = N))
    σ²new = vec(N ∈ reduce_dims ? σ² : mean(σ², dims = N))

    bn.μ = res_mtm .* bn.μ .+ mtm .* μnew
    bn.σ² = res_mtm .* bn.σ² .+ mtm .* (m / (m - one(V))) .* σ²new
    return nothing
end

ChainRulesCore.@non_differentiable _track_stats!(::Any...)

"""
    BatchNorm(channels::Integer, λ=identity;
              initβ=zeros32, initγ=ones32,
              affine = true, track_stats = true,
              ϵ=1f-5, momentum= 0.1f0)

[Batch Normalization](https://arxiv.org/abs/1502.03167) layer.
`channels` should be the size of the channel dimension in your data (see below).

Given an array with `N` dimensions, call the `N-1`th the channel dimension. For
a batch of feature vectors this is just the data dimension, for `WHCN` images
it's the usual channel dimension.

`BatchNorm` computes the mean and variance for each `D_1×...×D_{N-2}×1×D_N`
input slice and normalises the input accordingly.

If `affine=true`, it also applies  a shift and a rescale to the input
through to learnable per-channel bias β and scale γ parameters.

After normalisation, elementwise activation `λ` is applied.

If `track_stats=true`, accumulates mean and var statistics in training phase
that will be used to renormalize the input in test phase.

Use [`testmode!`](@ref) during inference.

# Examples

```julia
julia> using Statistics

julia> xs = rand(3, 3, 3, 2);  # a batch of 2 images, each having 3 channels

julia> m = BatchNorm(3);

julia> Flux.trainmode!(m);

julia> isapprox(std(m(xs)), 1, atol = 0.1) && std(xs) != std(m(xs))
true
```
"""
mutable struct BatchNorm{F, V, N, W}
    λ::F  # activation function
    β::V  # bias
    γ::V  # scale
    μ::W     # moving mean
    σ²::W    # moving var
    ϵ::N
    momentum::N
    affine::Bool
    track_stats::Bool
    active::Union{Bool, Nothing}
    chs::Int # number of channels
end

function BatchNorm(chs::Int, λ = identity;
                   initβ = zeros32, initγ = ones32,
                   affine = true, track_stats = true,
                   ϵ = 1.0f-5, momentum = 0.1f0)
    β = affine ? initβ(chs) : nothing
    γ = affine ? initγ(chs) : nothing
    μ = track_stats ? zeros32(chs) : nothing
    σ² = track_stats ? ones32(chs) : nothing

    return BatchNorm(λ, β, γ,
                     μ, σ², ϵ, momentum,
                     affine, track_stats,
                     nothing, chs)
end

@functor BatchNorm
trainable(bn::BatchNorm) = hasaffine(bn) ? (β = bn.β, γ = bn.γ) : (;)

function (BN::BatchNorm)(x)
    @assert size(x, ndims(x) - 1) == BN.chs
    N = ndims(x)
    reduce_dims = [1:(N - 2); N]
    affine_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return _norm_layer_forward(BN, x; reduce_dims, affine_shape)
end

function testmode!(m::BatchNorm, mode = true)
    return (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
end

function Base.show(io::IO, l::BatchNorm)
    print(io, "BatchNorm($(l.chs)")
    (l.λ == identity) || print(io, ", $(l.λ)")
    hasaffine(l) || print(io, ", affine=false")
    return print(io, ")")
end

"""
    InstanceNorm(channels::Integer, λ=identity;
                 initβ=zeros32, initγ=ones32,
                 affine=false, track_stats=false,
                 ϵ=1f-5, momentum=0.1f0)

[Instance Normalization](https://arxiv.org/abs/1607.08022) layer.
`channels` should be the size of the channel dimension in your data (see below).

Given an array with `N > 2` dimensions, call the `N-1`th the channel dimension.
For `WHCN` images it's the usual channel dimension.

`InstanceNorm` computes the mean and variance for each `D_1×...×D_{N-2}×1×1`
input slice and normalises the input accordingly.

If `affine=true`, it also applies  a shift and a rescale to the input
through to learnable per-channel bias `β` and scale `γ` parameters.

If `track_stats=true`, accumulates mean and var statistics in training phase
that will be used to renormalize the input in test phase.

**Warning**: the defaults for `affine` and `track_stats` used to be `true`
in previous Flux versions (< v0.12).

# Examples

```jldoctest
julia> using Statistics

julia> xs = rand(3, 3, 3, 2);  # a batch of 2 images, each having 3 channels

julia> m = InstanceNorm(3);

julia> y = m(xs);

julia> isapprox(std(y, dims = 1:2), ones(1, 1, 3, 2), atol = 0.2) &&
           std(y, dims = 1:2) != std(xs, dims = 1:2)
true
```
"""
mutable struct InstanceNorm{F, V, N, W}
    λ::F  # activation function
    β::V  # bias
    γ::V  # scale
    μ::W  # moving mean
    σ²::W  # moving var
    ϵ::N
    momentum::N
    affine::Bool
    track_stats::Bool
    active::Union{Bool, Nothing}
    chs::Int # number of channels
end

function InstanceNorm(chs::Int, λ = identity;
                      initβ = zeros32, initγ = ones32,
                      affine = false, track_stats = false,
                      ϵ = 1.0f-5, momentum = 0.1f0)
    if track_stats
        Base.depwarn("`track_stats=true` will be removed from InstanceNorm in Flux 0.14. The default value is `track_stats=false`, which will work as before.",
                     :InstanceNorm)
    end

    β = affine ? initβ(chs) : nothing
    γ = affine ? initγ(chs) : nothing
    μ = track_stats ? zeros32(chs) : nothing
    σ² = track_stats ? ones32(chs) : nothing

    return InstanceNorm(λ, β, γ,
                        μ, σ², ϵ, momentum,
                        affine, track_stats,
                        nothing, chs)
end

@functor InstanceNorm
trainable(in::InstanceNorm) = hasaffine(in) ? (β = in.β, γ = in.γ) : (;)

function (l::InstanceNorm)(x)
    @assert ndims(x) > 2
    @assert size(x, ndims(x) - 1) == l.chs
    N = ndims(x)
    reduce_dims = 1:(N - 2)
    affine_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return _norm_layer_forward(l, x; reduce_dims, affine_shape)
end

function testmode!(m::InstanceNorm, mode = true)
    return (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
end

function Base.show(io::IO, l::InstanceNorm)
    print(io, "InstanceNorm($(l.chs)")
    l.λ == identity || print(io, ", $(l.λ)")
    hasaffine(l) || print(io, ", affine=false")
    return print(io, ")")
end

"""
    GroupNorm(channels::Integer, G::Integer, λ=identity;
              initβ=zeros32, initγ=ones32,
              affine=true, track_stats=false,
              ϵ=1f-5, momentum=0.1f0)

[Group Normalization](https://arxiv.org/abs/1803.08494) layer.

`chs` is the number of channels, the channel dimension of your input.
For an array of N dimensions, the `N-1`th index is the channel dimension.

`G` is the number of groups along which the statistics are computed.
The number of channels must be an integer multiple of the number of groups.

`channels` should be the size of the channel dimension in your data (see below).

Given an array with `N > 2` dimensions, call the `N-1`th the channel dimension.
For `WHCN` images it's the usual channel dimension.

If `affine=true`, it also applies  a shift and a rescale to the input
through to learnable per-channel bias `β` and scale `γ` parameters.

If `track_stats=true`, accumulates mean and var statistics in training phase
that will be used to renormalize the input in test phase.

# Examples

```jldoctest
julia> using Statistics

julia> xs = rand(3, 3, 4, 2);  # a batch of 2 images, each having 4 channels

julia> m = GroupNorm(4, 2);

julia> y = m(xs);

julia> isapprox(std(y[:, :, 1:2, 1]), 1, atol = 0.1) &&
           std(xs[:, :, 1:2, 1]) != std(y[:, :, 1:2, 1])
true

julia> isapprox(std(y[:, :, 3:4, 2]), 1, atol = 0.1) &&
           std(xs[:, :, 3:4, 2]) != std(y[:, :, 3:4, 2])
true
```  # number of groups
```
"""
mutable struct GroupNorm{F, V, N, W}
    G::Int  # number of groups
    λ::F  # activation function
    β::V  # bias
    γ::V  # scale
    μ::W     # moving mean
    σ²::W    # moving std
    ϵ::N
    momentum::N
    affine::Bool
    track_stats::Bool
    active::Union{Bool, Nothing}
    chs::Int # number of channels
end

@functor GroupNorm
trainable(gn::GroupNorm) = hasaffine(gn) ? (β = gn.β, γ = gn.γ) : (;)

function GroupNorm(chs::Int, G::Int, λ = identity;
                   initβ = zeros32, initγ = ones32,
                   affine = true, track_stats = false,
                   ϵ = 1.0f-5, momentum = 0.1f0)
    if track_stats
        Base.depwarn("`track_stats=true` will be removed from GroupNorm in Flux 0.14. The default value is `track_stats=false`, which will work as before.",
                     :GroupNorm)
    end

    chs % G == 0 ||
        error("The number of groups ($(G)) must divide the number of channels ($chs)")

    β = affine ? initβ(chs) : nothing
    γ = affine ? initγ(chs) : nothing
    μ = track_stats ? zeros32(G) : nothing
    σ² = track_stats ? ones32(G) : nothing

    return GroupNorm(G, λ,
                     β, γ,
                     μ, σ²,
                     ϵ, momentum,
                     affine, track_stats,
                     nothing, chs)
end

function (gn::GroupNorm)(x)
    @assert ndims(x) > 2
    @assert size(x, ndims(x) - 1) == gn.chs
    N = ndims(x)
    sz = size(x)
    x = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ gn.G, gn.G, sz[N])
    N = ndims(x)
    reduce_dims = 1:(N - 2)
    affine_shape = ntuple(i -> i ∈ (N - 1, N - 2) ? size(x, i) : 1, N)
    x = _norm_layer_forward(gn, x; reduce_dims, affine_shape)
    return reshape(x, sz)
end

function testmode!(m::GroupNorm, mode = true)
    return (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
end

function Base.show(io::IO, l::GroupNorm)
    # print(io, "GroupNorm($(join(size(l.β), ", "))", ", ", l.G)
    print(io, "GroupNorm($(l.chs), $(l.G)")
    l.λ == identity || print(io, ", ", l.λ)
    hasaffine(l) || print(io, ", affine=false")
    return print(io, ")")
end

"""
    hasaffine(l)

Return `true` if a normalisation layer has trainable shift and
scale parameters, `false` otherwise.

See [`BatchNorm`](@ref), [`InstanceNorm`](@ref), [`GroupNorm`](@ref), and [`LayerNorm`](@ref).
"""
hasaffine(l::Union{BatchNorm, InstanceNorm, LayerNorm, GroupNorm}) = l.affine
