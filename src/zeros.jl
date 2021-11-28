import Base: +, -, *, /, reshape, broadcasted

"""
    Zeros()

Acts as a stand-in for an array of zeros that can be
used during training which is ignored by the optimisers.

Useful to turn bias off for a forward pass of a layer.

## Examples

```julia-repl
julia> bias_less_conv = Conv((2,2), 1=>3; bias = false)
Conv((2, 2), 1=>3)

julia> params(bias_less_conv) |> length
1

julia> bias_less_conv.bias
Flux.Zeros()
```
"""
struct Zeros end
# To allow for things like Dense(10, 2, initb = Zeros)
Zeros(args...) = Zeros()

Base.reshape(x::Zeros, dims...) = x

+(::Zeros, b::AbstractArray) = b
+(a::AbstractArray, ::Zeros) = a
+(a::Zeros, ::Zeros) = a

-(::Zeros, b::AbstractArray) = -b
-(a::AbstractArray, ::Zeros) = a
-(a::Zeros, ::Zeros) = a

# Some opportunities to avoid scalar indexing, intermediaries
# Since it replicates a little of what we expect Base to do,
# it should be possible to remove in the future, but for now,
# these help with performance.
broadcasted(::typeof(+), a::AbstractArray, b::Zeros) = a
broadcasted(::typeof(+), a::Zeros, b::AbstractArray) = b
broadcasted(::typeof(-), a::AbstractArray, b::Zeros) = a
broadcasted(::typeof(-), a::Zeros, b::AbstractArray) = -b
# Need adjoints for these or else the gradient w.r.t to the non-Zeros arg will be nothing as well
@adjoint broadcasted(::typeof(*), a::AbstractArray, b::Zeros) =
    zero(a), _ -> (nothing, zero(a), nothing)
@adjoint broadcasted(::typeof(*), a::Zeros, b::AbstractArray) =
    zero(b), _ -> (nothing, nothing, zero(b))
@adjoint broadcasted(::typeof(/), a::Zeros, b::AbstractArray) =
    zero(b), _ -> (nothing, nothing, zero(b))

# Pass-through for layer constructors
create_bias(weights::AbstractArray, bias::Flux.Zeros, dims::Integer...) = bias
