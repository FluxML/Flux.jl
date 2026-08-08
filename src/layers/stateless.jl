
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

# BFloat16 target: route through Flux's integer-domain conversion (`_to_bf16`), since
# the native `convert(_, BFloat16)` crashes on some LLVM versions (JuliaMath/BFloat16s.jl#107).
_match_eltype(layer, ::Type{BFloat16}, x::AbstractArray{<:Union{AbstractFloat, Integer}}) = _to_bf16(x)

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
should be preferred to this method existing 
only for backward compatibility.
"""
flatten(x) = MLUtils.flatten(x)

"""
    normalise(x; dims=ndims(x), eps=1f-5)

Same as [`NNlib.normalise`](@ref), to which this method forwards.
Kept for backward compatibility; prefer `NNlib.normalise` in new code.
"""
normalise(x::AbstractArray; kw...) = NNlib.normalise(x; kw...)
