
"""
    normalise(x; dims=ndims(x), eps=1e-5)

Normalise `x` to mean 0 and standard deviation 1 across the dimension(s) given by `dims`.
Per default, `dims` is the last dimension. 
`eps` is a small term added to the denominator for numerical stability.

# Examples
```jldoctest
julia> using Statistics

julia> x = [90, 100, 110, 130, 70];

julia> mean(x), std(x; corrected=false)
(100.0, 20.0)

julia> y = Flux.normalise(x)
5-element Vector{Float64}:
 -0.49999975000012503
  0.0
  0.49999975000012503
  1.499999250000375
 -1.499999250000375

julia> isapprox(std(y; corrected=false), 1, atol=1e-5)
true

julia> x = rand(10:100, 10, 10);

julia> y = Flux.normalise(x, dims=1);

julia> isapprox(std(y; dims=1, corrected=false), ones(1, 10), atol=1e-5)
true
```
"""
@inline function normalise(x::AbstractArray; dims=ndims(x), eps=ofeltype(x, 1e-5), ϵ=nothing)
  μ = mean(x, dims=dims)
  σ = std(x, dims=dims, mean=μ, corrected=false)
  return @. (x - μ) / (σ + eps)
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
# NB using _eltype gets Float64 from Tracker.TrackedArray{Float64}, not TrackedReal
_match_eltype(layer, x) = _match_eltype(layer, _eltype(layer.weight), x)

# Trivial rule:
function ChainRulesCore.rrule(::typeof(_match_eltype), layer, ::Type{T}, x::AbstractArray) where {T}
  _match_eltype(layer, T, x), dx -> (NoTangent(), ZeroTangent(), NoTangent(), dx)  # does not un-thunk dx
end
function ChainRulesCore.rrule(::typeof(_match_eltype), layer, x::AbstractArray)
  _match_eltype(layer, x), dx -> (ZeroTangent(), NoTangent(), dx)  # does not un-thunk dx
end

# We have to define our own flatten in order 
# to load previously saved models. 
# See #2195 #2204
"""
  flatten(x)

Same as [`MLUtils.flatten`](@ref), which 
should be prefered to this method existing 
only for backward compatibility.
"""
flatten(x) = MLUtils.flatten(x)
