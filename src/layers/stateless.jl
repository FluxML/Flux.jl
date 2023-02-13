"""
    normalise(x; dims=ndims(x), ϵ=1e-5)

Normalise `x` to mean 0 and standard deviation 1 across the dimension(s) given by `dims`.
Per default, `dims` is the last dimension. 
`ϵ` is a small additive factor added to the denominator for numerical stability.

# Examples
```jldoctest
julia> using Statistics

julia> x = [9, 10, 20, 60];

julia> y = Flux.normalise(x);

julia> isapprox(std(y), 1, atol=0.2) && std(y) != std(x)
true

julia> x = rand(1:100, 10, 2);

julia> y = Flux.normalise(x, dims=1);

julia> isapprox(std(y, dims=1), ones(1, 2), atol=0.2) && std(y, dims=1) != std(x, dims=1)
true
```
"""
@inline function normalise(x::AbstractArray; dims=ndims(x), ϵ=ofeltype(x, 1e-5))
  μ = mean(x, dims=dims)
  σ = std(x, dims=dims, mean=μ, corrected=false)
  return @. (x - μ) / (σ + ϵ)
end

"""
    _match_eltype(layer, ::Type{T}, x)
    _match_eltype(layer, x)

This internal function corrects most layer input to match the type of the weights.
The second method uses `T = eltype(layer.weight)`.

It solves a common performance bug: Before, accidentally supplying `Float64` input,
or an activation function which produces `Float64`, would silently run the
entire forward pass in this precision.
"""
_match_eltype(layer, ::Type{T}, x::AbstractArray{T}) where {T} = x

# A common mistake, print a friendly warning, and fix it:
function _match_eltype(layer, ::Type{Float32}, x::AbstractArray{Float64})
  # This warning is the only reason this needs to take the layer.
  @warn "Layer with Float32 parameters got Float64 input.
  The input will be converted, but any earlier layers may be very slow." layer summary(x) maxlog=1
  convert(AbstractArray{Float32}, x)
end

# Bug in Float16 use?
function _match_eltype(layer, ::Type{Float16}, x::AbstractArray{Float32})
  @warn "Layer with Float16 parameters got Float32 input.
  The input will be converted, but may indicate a problem in earlier layers." layer summary(x) maxlog=1
  convert(AbstractArray{Float16}, x)
end

# Allow OneHot to reach specialisation of * etc:
_match_eltype(layer, ::Type, x::OneHotLike) = x

# Other floats, and integers, silently fix.
function _match_eltype(layer, ::Type{T}, x::AbstractArray{<:Union{AbstractFloat, Integer}}) where {T}
  convert(AbstractArray{T}, x)
end

# Weird types like Nil, Dual, etc, we allow through:
_match_eltype(layer, ::Type, x::AbstractArray) = x

# 2-arg method, for common layers with layer.weight
_match_eltype(layer, x) = _match_eltype(layer, eltype(layer.weight), x)

# Trivial rule:
function ChainRulesCore.rrule(::typeof(_match_eltype), layer, ::Type{T}, x::AbstractArray) where {T}
  _match_eltype(layer, T, x), dx -> (NoTangent(), ZeroTangent(), NoTangent(), dx)  # does not un-thunk dx
end
function ChainRulesCore.rrule(::typeof(_match_eltype), layer, x::AbstractArray)
  _match_eltype(layer, x), dx -> (ZeroTangent(), NoTangent(), dx)  # does not un-thunk dx
end

