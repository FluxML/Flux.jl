import Base: +, -, *, reshape, size
import Base.Broadcast: broadcasted, Broadcasted, BroadcastStyle
import Zygote: unbroadcast

"""
    Zeros()
    Zeros(size...)
    Zeros(Type, size...)

Acts as a stand-in for an array of zeros that can be
used during training which is ignored by the optimisers.

Useful to turn bias off for a forward pass of a layer.

## Examples

```julia
julia> Flux.Zeros(3,3)
3×3 Flux.Zeros{Bool,2}:
 false  false  false
 false  false  false
 false  false  false

julia> Flux.Zeros(Float32, 3,3)
3×3 Flux.Zeros{Float32,2}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> rand(3,3) .+ Flux.Zeros()
3×3 Array{Float64,2}:
 0.198739  0.490459  0.785386
 0.779074  0.39986   0.66383
 0.854981  0.447292  0.314497

julia> bias_less_conv = Conv((2,2), 1=>3, bias = Flux.Zeros())
Conv((2, 2), 1=>3)
```
"""
struct Zeros{T,N} <: AbstractArray{T,N}
  size::Tuple
end

Zeros(::Type{T}, sz...) where T = Zeros{T,length(sz)}(sz)
Zeros(sz::Integer...) = Zeros(Bool, sz...)

Base.size(xs::Zeros) = xs.size
Base.axes(xs::Zeros) = Base.OneTo.(size(xs))

Base.IndexStyle(::Type{<:Zeros}) = IndexLinear()

Base.getindex(xs::Zeros{T,N}, I::Int) where {T,N} = zero(T)
Base.getindex(xs::Zeros{T,N}, inds::Union{Base.OneTo, Base.UnitRange}) where {T,N} =
              Zeros(T, length(inds))

Base.collect(xs::Zeros{T,N}) where {T,N} = fill(zero(T), size(xs))

# Or else they'll turn into a ReshapedArray and all the stuff below is circumvented
Base.reshape(xs::Zeros{T}, dims::Vararg{Union{Colon, Int64},N}) where {T, N} = Zeros(T, Base._reshape_uncolon(xs, dims)...)
@adjoint reshape(xs::Zeros{T}, dims...) where T =
                reshape(xs, dims...), _ -> nothing

# Define basic ops
for f in (:+, :-)
  @eval @inline function $f(a::Union{AbstractArray{<:Number}, Zeros}, b::Zeros)
    @assert size(a) == size(b) throw(DimensionMismatch("dimensions must match"))
    a
  end
end

+(a::Zeros, b::AbstractArray) = b + a
-(a::Zeros, b::AbstractArray) = -b + a

Base.copy(xs::Zeros{T,N}) where {T,N} = xs

Base.copyto!(dest::AbstractArray, ::Zeros) = copyto!(dest, zeros(size(dest)))
Base.copyto!(dest::Zeros, ::AbstractArray) = dest
Base.copyto!(dest::Zeros, ::Zeros) = dest

# Define broadcasting behaviour
for op in (:+, :-)
  @eval function broadcasted(::typeof($op), a::AbstractArray{<:Number}, b::Zeros)
    bs = Broadcast.broadcast_shape(size(a), size(b))
    size(a) == bs && return a
    sz = similar(a, bs)
    sz .= a
    return sz
  end
end

# This is a bit of a whack-a-mole with avoiding ambiguity while still making sure we capture all signatures...

@adjoint broadcasted(::typeof(+), a::AbstractArray{<:Number}, b::Zeros) = broadcasted(+, a, b), ā -> (nothing, unbroadcast(a, ā), nothing)
@adjoint broadcasted(::typeof(+), a::Zeros, b::AbstractArray{<:Number}) = broadcasted(+, b, a), b̄ -> (nothing, nothing, unbroadcast(b, b̄))

@adjoint broadcasted(::typeof(-), a::Zeros, b::AbstractArray{<:Number}) = broadcasted(+, -b, a), b̄ -> (nothing, nothing, -unbroadcast(b, b̄))
@adjoint broadcasted(::typeof(-), a::AbstractArray{<:Number}, b::Zeros) = broadcasted(+, a, b), ā -> (nothing, unbroadcast(a, ā), nothing)

@adjoint function broadcasted(::typeof(*), a::AbstractArray{T}, b::Zeros) where T<:Number 
  zs = zeros(T, Broadcast.broadcast_shape(size(a), size(b))...)
  zs, ā -> (nothing, unbroadcast(a, zs), nothing)
end

@adjoint function broadcasted(::typeof(*), a::Zeros, b::AbstractArray{T}) where T<:Number 
  zs = zeros(T, Broadcast.broadcast_shape(size(a), size(b))...)
  zs, b̄ -> (nothing, nothing, unbroadcast(b, zs))
end

for op in (:+, :-, :*)
  @eval @adjoint broadcasted(::typeof($op), a::Zeros, b::Zeros) = Zeros(Broadcast.broadcast_shape(size(a), size(b))...), _ -> (nothing, nothing, nothing)
  # To avoid ambiguity with 0-size Zeros below
  @eval @adjoint broadcasted(::typeof($op), a::Zeros, b::Zeros{T, 0}) where T<:Number = a, _ -> (nothing, nothing, nothing)
  @eval @adjoint broadcasted(::typeof($op), a::Zeros{T, 0}, b::Zeros{T, 0}) where T<:Number = a, _ -> (nothing, nothing, nothing)
  @eval @adjoint broadcasted(::typeof($op), a::Zeros{T, 0}, b::Zeros) where T<:Number = a, _ -> (nothing, nothing, nothing)
end

# Some opportunities to avoid scalar indexing, intermediaries
# Since it replicates a little of what we expect Base to do,
# it should be possible to remove in the future, but for now,
# these help with performance.
@adjoint broadcasted(::typeof(+), a::AbstractArray{T}, b::Zeros{<:Number,0}) where T<:Number = a, ā -> (nothing, unbroadcast(a, ā), nothing)
@adjoint broadcasted(::typeof(+), a::Zeros{T,0}, b::AbstractArray{<:Number}) where T<:Number = b, b̄ -> (nothing, nothing, unbroadcast(b, b̄))
@adjoint broadcasted(::typeof(-), a::AbstractArray{<:Number}, b::Zeros{T,0}) where T<:Number = a, ā -> (nothing, unbroadcast(a, ā), nothing)
@adjoint broadcasted(::typeof(-), a::Zeros{T,0}, b::AbstractArray{<:Number}) where T<:Number = -b, b̄ -> (nothing, nothing, -unbroadcast(b, b̄))
@adjoint broadcasted(::typeof(*), a::AbstractArray{<:Number}, b::Zeros{T,0}) where T<:Number = zero(a), ā -> (nothing, unbroadcast(a, Zeros(eltype(a), size(a)...)), nothing)
@adjoint broadcasted(::typeof(*), a::Zeros{T,0}, b::AbstractArray{<:Number}) where T<:Number = zero(b), b̄ -> (nothing, nothing, unbroadcast(b, Zeros(eltype(b), size(b)...)))
@adjoint broadcasted(::typeof(/), a::Zeros{T,0}, b::AbstractArray{<:Number}) where T<:Number = zero(b), b̄ -> (nothing, nothing, unbroadcast(b, Zeros(eltype(b), size(b)...)))
