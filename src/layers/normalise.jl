# Internal function, used only for layers defined in this file.
_isactive(m, x) = isnothing(m.active) ? NNlib.within_gradient(x) : m.active

# Internal function, used only in this file.
_tidy_active(mode::Bool) = mode
_tidy_active(::Nothing) = nothing
_tidy_active(mode) = mode === :auto ? nothing : throw(ArgumentError(lazy"active = $(repr(mode)) is not accepted, must be true/false/nothing or :auto"))

"""
    Dropout(p; [dims, rng, active])

Layer implementing [dropout](https://arxiv.org/abs/1207.0580) with the given probability.
This is used as a regularisation, i.e. to reduce overfitting.

While training, it sets each input to `0` (with probability `p`)
or else scales it by `1 / (1 - p)`, using the [`NNlib.dropout`](@ref) function.
While testing, it has no effect.

By default the mode will switch automatically, but it can also
be controlled manually via [`Flux.testmode!`](@ref),
or by passing keyword `active=true` for training mode.

By default every input is treated independently. With the `dims` keyword,
instead it takes a random choice only along that dimension.
For example `Dropout(p; dims = 3)` will randomly zero out entire channels on WHCN input
(also called 2D dropout).

Keyword `rng` lets you specify a custom random number generator.
(Only supported on the CPU.)

# Examples
```julia-repl
julia> m = Chain(Dense(ones(3,2)), Dropout(0.4))
Chain(
  Dense(2 => 3),                        # 9 parameters
  Dropout(0.4),
)

julia> m(ones(2, 7))  # test mode, no effect
3×7 Matrix{Float64}:
 2.0  2.0  2.0  2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0  2.0  2.0  2.0

julia> Flux.trainmode!(m)  # equivalent to use within gradient
Chain(
  Dense(2 => 3),                        # 9 parameters
  Dropout(0.4, active=true),
)

julia> m(ones(2, 7))
3×7 Matrix{Float64}:
 0.0      0.0      3.33333  0.0      0.0      0.0  0.0
 3.33333  0.0      3.33333  0.0      3.33333  0.0  3.33333
 3.33333  3.33333  0.0      3.33333  0.0      0.0  3.33333

julia> y = m(ones(2, 10_000));

julia> using Statistics

julia> mean(y)  # is about 2.0, same as in test mode
1.9989999999999961

julia> mean(iszero, y)  # is about 0.4
0.4003
```
"""
mutable struct Dropout{F<:Real,D,R<:AbstractRNG}
  p::F
  dims::D
  active::Union{Bool, Nothing}
  rng::R
end
Dropout(p::Real, dims, active) = Dropout(p, dims, active, default_rng())

function Dropout(p::Real; dims=:, active::Union{Bool,Nothing} = nothing, rng = default_rng())
  0 ≤ p ≤ 1 || throw(ArgumentError(lazy"Dropout expects 0 ≤ p ≤ 1, got p = $p"))
  Dropout(p, dims, active, rng)
end

@layer Dropout trainable=()

(a::Dropout)(x) = dropout(a.rng, x, a.p * _isactive(a, x); dims=a.dims)

testmode!(m::Dropout, mode=true) =
  (m.active = isnothing(_tidy_active(mode)) ? nothing : !mode; m)

function Base.show(io::IO, d::Dropout)
  print(io, "Dropout(", d.p)
  d.dims != (:) && print(io, ", dims=", d.dims)
  d.active == nothing || print(io, ", active=", d.active)
  print(io, ")")
end

"""
    AlphaDropout(p; [rng, active])

A dropout layer. Used in
[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515).
The AlphaDropout layer ensures that mean and variance of activations
remain the same as before.

Does nothing to the input once [`testmode!`](@ref) is true.

# Examples
```jldoctest
julia> using Statistics

julia> x = randn32(1000,1);

julia> m = Chain(Dense(1000 => 1000, selu), AlphaDropout(0.2));

julia> Flux.trainmode!(m);

julia> y = m(x);

julia> isapprox(std(x), std(y), atol=0.2)
true
```
"""
mutable struct AlphaDropout{F,R<:AbstractRNG}
  p::F
  active::Union{Bool, Nothing}
  rng::R
end

AlphaDropout(p, active) = AlphaDropout(p, active, default_rng())
function AlphaDropout(p; rng = default_rng(), active::Union{Bool,Nothing} = nothing)
  0 ≤ p ≤ 1 || throw(ArgumentError(lazy"AlphaDropout expects 0 ≤ p ≤ 1, got p = $p"))
  AlphaDropout(p, active, rng)
end

@layer AlphaDropout trainable=()

function (a::AlphaDropout)(x::AbstractArray{T}) where T
  _isactive(a, x) || return x
  p = a.p
  iszero(p) && return x
  isone(p) && return sign.(x) .* T(0)

  α′ = T(-1.7580993408473766) # selu(-Inf) == -λα
  A = T(inv(sqrt((1 - p) * (1 + p * α′^2))))
  B = T(-A * α′ * p)

  noise = rand!(a.rng, similar(x))
  return A .* ifelse.(noise .> p, x, α′) .+ B
end

testmode!(m::AlphaDropout, mode=true) =
  (m.active = isnothing(_tidy_active(mode)) ? nothing : !mode; m)

Base.show(io::IO, d::AlphaDropout) = print(io, "AlphaDropout(", d.p, ")")

"""
    LayerNorm(size..., λ=identity; affine=true, eps=1f-5)

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

julia> isapprox(std(y, dims=1:3), ones(1, 1, 1, 2), atol=0.1) && std(y, dims=1:3) != std(xs, dims=1:3)
true
```
"""
struct LayerNorm{F,D,T,N}
  λ::F
  diag::D
  ϵ::T
  size::NTuple{N,Int}
  affine::Bool
end

function LayerNorm(size::Tuple{Vararg{Int}}, λ=identity; affine::Bool=true, eps::Real=1f-5)
  diag = affine ? Scale(size..., λ) : λ!=identity ? Base.Fix1(broadcast, λ) : identity
  return LayerNorm(λ, diag, eps, size, affine)
end
LayerNorm(size::Integer...; kw...) = LayerNorm(Int.(size); kw...)
LayerNorm(size_act...; kw...) = LayerNorm(Int.(size_act[1:end-1]), size_act[end]; kw...)

@layer :noexpand LayerNorm

function (a::LayerNorm)(x::AbstractArray)
  ChainRulesCore.@ignore_derivatives if a.diag isa Scale
    for d in 1:ndims(a.diag.scale)
      _size_check(a, x, d => size(a.diag.scale, d))
    end
  end
  eps = convert(float(eltype(x)), a.ϵ)  # avoids promotion for Float16 data, but should ε chage too?
  a.diag(normalise(x; dims=1:length(a.size), eps))
end

function Base.show(io::IO, l::LayerNorm)
  print(io, "LayerNorm(", join(l.size, ", "))
  l.λ === identity || print(io, ", ", l.λ)
  hasaffine(l) || print(io, ", affine=false")
  print(io, ")")
end

# For InstanceNorm, GroupNorm, and BatchNorm.
# Compute the statistics on the slices specified by reduce_dims.
# reduce_dims=[1,...,N-2,N] for BatchNorm
# reduce_dims=[1,...,N-2] for InstanceNorm and GroupNorm
function _norm_layer_forward(
  l, x::AbstractArray{T, N}; reduce_dims, affine_shape,
) where {T, N}
  if !_isactive(l, x) && l.track_stats # testmode with tracked stats
    stats_shape = ChainRulesCore.ignore_derivatives() do
      ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
    end
    μ = reshape(l.μ, stats_shape)
    σ² = reshape(l.σ², stats_shape)
  else # trainmode or testmode without tracked stats
    μ = mean(x; dims=reduce_dims)
    σ² = var(x; mean=μ, dims=reduce_dims, corrected=false)
    if l.track_stats
      _track_stats!(l, x, μ, σ², reduce_dims) # update moving mean/std
    end
  end

  eps = convert(float(T), l.ϵ)
  hasaffine(l) || return l.λ.(_norm_layer_forward(x, μ, σ², eps))

  γ = reshape(l.γ, affine_shape)
  β = reshape(l.β, affine_shape)

  scale = γ ./ sqrt.(σ² .+ eps)
  bias = .-scale .* μ .+ β
  l.λ.(scale .* x .+ bias)
end

@inline _norm_layer_forward(x, μ, σ², ϵ) = (x .- μ) ./ sqrt.(σ² .+ ϵ)

function _track_stats!(
  bn, x::AbstractArray{T, N}, μ, σ², reduce_dims,
) where {T, N}
  V = eltype(bn.σ²)
  mtm = bn.momentum
  res_mtm = one(V) - mtm
  m = prod(size(x, i) for i in reduce_dims)

  μnew = vec(N ∈ reduce_dims ? μ : mean(μ, dims=N))
  σ²new = vec(N ∈ reduce_dims ? σ² : mean(σ², dims=N))

  # ForwardDiff.value removes Dual, was an error, issue #2122
  bn.μ .= value.(res_mtm .* bn.μ .+ mtm .* μnew)
  bn.σ² .= value.(res_mtm .* bn.σ² .+ mtm .* (m / (m - one(V))) .* σ²new)
  return nothing
end

ChainRulesCore.@non_differentiable _track_stats!(::Any...)

"""
    BatchNorm(channels::Integer, λ=identity;
              initβ=zeros32, initγ=ones32,
              affine=true, track_stats=true, active=nothing,
              eps=1f-5, momentum= 0.1f0)

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
```julia-repl
julia> using Statistics

julia> xs = rand(3, 3, 3, 2);  # a batch of 2 images, each having 3 channels

julia> m = BatchNorm(3);

julia> Flux.trainmode!(m);

julia> isapprox(std(m(xs)), 1, atol=0.1) && std(xs) != std(m(xs))
true
```
"""
mutable struct BatchNorm{F,V,N,W}
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

function BatchNorm(chs::Int, λ=identity;
          initβ=zeros32, initγ=ones32,
          affine::Bool=true, track_stats::Bool=true, active::Union{Bool,Nothing}=nothing,
          eps::Real=1f-5, momentum::Real=0.1f0)


  β = affine ? initβ(chs) : nothing
  γ = affine ? initγ(chs) : nothing
  μ = track_stats ? zeros32(chs) : nothing
  σ² = track_stats ? ones32(chs) : nothing

  return BatchNorm(λ, β, γ,
            μ, σ², eps, momentum,
            affine, track_stats,
            active, chs)
end

@layer BatchNorm trainable=(β,γ)

function (BN::BatchNorm)(x::AbstractArray{T,N}) where {T,N}
  _size_check(BN, x, N-1 => BN.chs)
  reduce_dims = [1:N-2; N]
  affine_shape = ChainRulesCore.ignore_derivatives() do
    ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
  end
  return _norm_layer_forward(BN, x; reduce_dims, affine_shape)
end

testmode!(m::BatchNorm, mode=true) =
  (m.active = isnothing(_tidy_active(mode)) ? nothing : !mode; m)

function Base.show(io::IO, l::BatchNorm)
  print(io, "BatchNorm($(l.chs)")
  (l.λ == identity) || print(io, ", $(l.λ)")
  hasaffine(l) || print(io,  ", affine=false")
  l.active == nothing || print(io, ", active=", l.active)
  print(io, ")")
end


"""
    InstanceNorm(channels::Integer, λ=identity;
                 initβ=zeros32, initγ=ones32,
                 affine=false, track_stats=false,
                 eps=1f-5, momentum=0.1f0)

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

julia> isapprox(std(y, dims=1:2), ones(1, 1, 3, 2), atol=0.2) && std(y, dims=1:2) != std(xs, dims=1:2)
true
```
"""
mutable struct InstanceNorm{F,V,N,W}
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

function InstanceNorm(chs::Int, λ=identity;
                    initβ=zeros32, initγ=ones32,
                    affine::Bool=false, track_stats::Bool=false, active::Union{Bool,Nothing}=nothing,
                    eps::Real=1f-5, momentum::Real=0.1f0)

  β = affine ? initβ(chs) : nothing
  γ = affine ? initγ(chs) : nothing
  μ = track_stats ? zeros32(chs) : nothing
  σ² = track_stats ? ones32(chs) : nothing

  return InstanceNorm(λ, β, γ,
            μ, σ², eps, momentum,
            affine, track_stats,
            active, chs)
end

@layer InstanceNorm trainable=(β,γ)

function (l::InstanceNorm)(x::AbstractArray{T,N}) where {T,N}
  _size_check(l, x, N-1 => l.chs)
  reduce_dims = 1:N-2
  affine_shape = ChainRulesCore.ignore_derivatives() do
    ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
  end
  return _norm_layer_forward(l, x; reduce_dims, affine_shape)
end

testmode!(m::InstanceNorm, mode=true) =
  (m.active = isnothing(_tidy_active(mode)) ? nothing : !mode; m)

function Base.show(io::IO, l::InstanceNorm)
  print(io, "InstanceNorm($(l.chs)")
  l.λ == identity || print(io, ", $(l.λ)")
  hasaffine(l) || print(io,  ", affine=false")
  l.active == nothing || print(io, ", active=", l.active)
  print(io, ")")
end

"""
    GroupNorm(channels::Int, G::Int, λ = identity;
              initβ = zeros32,
              initγ = ones32,
              affine = true,
              eps = 1f-5,
              momentum = 0.1f0)

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


# Examples
```jldoctest
julia> using Statistics

julia> xs = rand(3, 3, 4, 2);  # a batch of 2 images, each having 4 channels

julia> m = GroupNorm(4, 2);

julia> y = m(xs);

julia> isapprox(std(y[:, :, 1:2, 1]), 1, atol=0.1) && std(xs[:, :, 1:2, 1]) != std(y[:, :, 1:2, 1])
true

julia> isapprox(std(y[:, :, 3:4, 2]), 1, atol=0.1) && std(xs[:, :, 3:4, 2]) != std(y[:, :, 3:4, 2])
true
```
"""
mutable struct GroupNorm{F,V,N}
  G::Int  # number of groups
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::Nothing     # moving mean
  σ²::Nothing    # moving std
  ϵ::N
  momentum::N
  affine::Bool
  track_stats::Bool
  active::Union{Bool, Nothing}
  chs::Int # number of channels
end

@layer GroupNorm trainable=(β,γ)

function GroupNorm(chs::Int, G::Int, λ=identity;
              initβ=zeros32, initγ=ones32,
              affine::Bool=true, active::Union{Bool,Nothing}=nothing,
              eps::Real=1f-5, momentum::Real=0.1f0)

  chs % G == 0 || error("The number of groups ($(G)) must divide the number of channels ($chs)")

  β = affine ? initβ(chs) : nothing
  γ = affine ? initγ(chs) : nothing
  μ = nothing
  σ² = nothing
  track_stats = false

  return GroupNorm(G, λ,
            β, γ,
            μ, σ²,
            eps, momentum,
            affine, track_stats,
            active, chs)
end

function (gn::GroupNorm)(x::AbstractArray)
  _size_check(gn, x, ndims(x)-1 => gn.chs)
  sz = size(x)
  x2 = reshape(x, sz[1:end-2]..., sz[end-1]÷gn.G, gn.G, sz[end])
  N = ndims(x2)  # == ndims(x)+1
  reduce_dims = 1:N-2
  affine_shape = ChainRulesCore.ignore_derivatives() do
    ntuple(i -> i ∈ (N-1, N-2) ? size(x2, i) : 1, N)
  end
  x3 = _norm_layer_forward(gn, x2; reduce_dims, affine_shape)
  return reshape(x3, sz)
end

testmode!(m::GroupNorm, mode = true) =
  (m.active = isnothing(_tidy_active(mode)) ? nothing : !mode; m)

function Base.show(io::IO, l::GroupNorm)
  # print(io, "GroupNorm($(join(size(l.β), ", "))", ", ", l.G)
  print(io, "GroupNorm($(l.chs), $(l.G)")
  l.λ == identity || print(io, ", ", l.λ)
  hasaffine(l) || print(io,  ", affine=false")
  l.active == nothing || print(io, ", active=", l.active)
  print(io, ")")
end

"""
    hasaffine(l)

Return `true` if a normalisation layer has trainable shift and
scale parameters, `false` otherwise.

See [`BatchNorm`](@ref), [`InstanceNorm`](@ref), [`GroupNorm`](@ref), and [`LayerNorm`](@ref).
"""
hasaffine(l::Union{BatchNorm, InstanceNorm, LayerNorm, GroupNorm}) = l.affine

struct WeightNorm{L, G, D}
    layer::L
    g::G

    which::Symbol
    dims::D
end
@layer WeightNorm

"""
    WeightNorm(layer::L, which::Symbol = :weight; dims = -1)

Apply weight normalization to a parameter given by `which` in a `layer`.

``w = g \\frac{\\mathbf{v}}{\\lVert \\mathbf{v} \\rVert}``

Decouples the magnitude of a weight tensor from its direction.
By default, normalization is applied along the output channel `dim=-1`
(equivalent to `dims=ndims(w)`).

### Example

```jldoctest
julia> c = Conv((3,), 1 => 2);

julia> wc = WeightNorm(c, :weight)
WeightNorm(
  Conv((3,), 1 => 2),                   # 8 parameters
  3×1×1 Array{Float32,...},             # 3 parameters
  :weight,
  3,
)                   # Total: 3 arrays, 11 parameters, 276 bytes.

julia> x = ones(Float32, 12, 1, 1);

julia> c(x) ≈ wc(x) # forward pass is the same as with the original layer
true
```

# Reference

Salimans & Kingma, _Weight Normalization_ (2016) <https://arxiv.org/abs/1602.07868>
"""
function WeightNorm(layer::L, which::Symbol = :weight; dims = -1) where L
    hasfield(L, which) || throw(ArgumentError("`$L` does not have field `:$which`."))

    x = getfield(layer, which)
    iszero(x) && throw(ArgumentError(
        "`$which` field for `$(typeof(layer))` is all zero, which will result in NaN."))

    d = if dims isa Colon
        1:ndims(x)
    elseif dims == -1
        dims = ndims(x)
    else
        dims
    end

    g = sqrt.(sum(abs2, x; dims) .+ eps(eltype(x)))
    WeightNorm(layer, g, which, dims)
end

(w::WeightNorm)(x) = reparametrize(w)(x)

"""
    reparametrize(wn::WeightNorm)

Apply `WeightNorm` reparametrization and return underlying `layer`.
"""
function reparametrize(wn::WeightNorm)
    ϵ = eps(eltype(wn.g))
    v = getfield(wn.layer, wn.which)
    n2 = sum(abs2, v; wn.dims)
    w = @. wn.g * v / sqrt(n2 + ϵ)

    fields, ctor = Functors.functor(wn.layer)
    return ctor(merge(
        fields, NamedTuple{(wn.which,)}((w,)),
    ))
end

function Base.show(io::IO, w::WeightNorm)
    print(io, "WeightNorm(")
    Base.show(io, w.layer)
    print(io, ", :", w.which, "; dims=", w.dims)
    print(io, ")")
end

"""
    remove_weight_norms(x)

Remove any [WeightNorm](@ref) parametrization in the model.

### Example

```jldoctest

julia> model = Chain(
           WeightNorm(Conv((3,), 1 => 2), :weight),
           WeightNorm(Conv((3,), 2 => 2), :weight),
       )
Chain(
  WeightNorm(
    Conv((3,), 1 => 2),                 # 8 parameters
    3×1×1 Array{Float32,...},           # 3 parameters
    :weight,
    3,
  ),
  WeightNorm(
    Conv((3,), 2 => 2),                 # 14 parameters
    3×2×1 Array{Float32,...},           # 6 parameters
    :weight,
    3,
  ),
)                   # Total: 6 arrays, 31 parameters, 588 bytes.

julia> Flux.remove_weight_norms(model)
Chain(
  Conv((3,), 1 => 2),                   # 8 parameters
  Conv((3,), 2 => 2),                   # 14 parameters
)                   # Total: 4 arrays, 22 parameters, 392 bytes.
```
"""
remove_weight_norms(x) = fmap(reparametrize, x; exclude=l -> l isa WeightNorm)
