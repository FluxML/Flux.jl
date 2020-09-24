module NilNumber

using LinearAlgebra


"""
    Nil <: Number
Nil is a singleton type with a single instance `nil`. Unlike
`Nothing` and `Missing` it subtypes `Number`.
"""
struct Nil <: Number end

const nil = Nil()

Nil(::T) where T<:Number = nil

Base.float(::Type{Nil}) = Nil
Base.copy(::Nil) = nil
Base.abs2(::Nil) = nil
Base.sqrt(::Nil) = nil
Base.zero(::Type{Nil}) = nil
Base.one(::Type{Nil}) = nil

Base.:+(::Nil) = nil
Base.:-(::Nil) = nil

Base.:+(::Nil, ::Nil) = nil
Base.:+(::Nil, ::Number) = nil
Base.:+(::Number, ::Nil) = nil

Base.:-(::Nil, ::Nil) = nil
Base.:-(::Nil, ::Number) = nil
Base.:-(::Number, ::Nil) = nil

Base.:*(::Nil, ::Nil) = nil
Base.:*(::Nil, ::Number) = nil
Base.:*(::Number, ::Nil) = nil

Base.:/(::Nil, ::Nil) = nil
Base.:/(::Nil, ::Number) = nil
Base.:/(::Number, ::Nil) = nil

Base.inv(::Nil) = nil

Base.isless(::Nil, ::Nil) = true
Base.isless(::Nil, ::Number) = true
Base.isless(::Number, ::Nil) = true

Base.abs(::Nil) = nil
Base.exp(::Nil) = nil

Base.typemin(::Type{Nil}) = nil
Base.typemax(::Type{Nil}) = nil
Base.:^(::Nil, ::Nil) = nil

# TODO: can this be shortened?
Base.promote(x::Nil, y::Nil) = (nil, nil)
Base.promote(x::Nil, y) = (nil, nil)
Base.promote(x, y::Nil) = (nil, nil)
Base.promote(x::Nil, y, z) = (nil, nil, nil)
Base.promote(x, y::Nil, z) = (nil, nil, nil)
Base.promote(x, y, z::Nil) = (nil, nil, nil)
Base.promote(x::Nil, y, z::Nil) = (nil, nil, nil)
Base.promote(x::Nil, y::Nil, z::Nil) = (nil, nil, nil)
Base.promote(x::Nil, y::Nil, z) = (nil, nil, nil)


LinearAlgebra.adjoint(::Nil) = nil
LinearAlgebra.transpose(::Nil) = nil

end  # module

using .NilNumber: Nil, nil

"""
    outdims(m, isize)

Calculate the output size of module `m` given an input of size `isize`.
`isize` should include the batch dimension.

Should work for all custom layers.
"""
outdims(m, isize) = with_logger(NullLogger()) do
    size(m(fill(nil, isize)))
end


## fixes for layers that don't work out of the box

for (fn, Dims) in ((:conv, DenseConvDims), (:depthwiseconv, DepthwiseConvDims))
    @eval begin
        function NNlib.$fn(a::AbstractArray{<:Real}, b::AbstractArray{Nil}, dims::$Dims) where T
            NNlib.$fn(fill(nil, size(a)), b, dims)
        end

        function NNlib.$fn(a::AbstractArray{Nil}, b::AbstractArray{<:Real}, dims::$Dims) where T
            NNlib.$fn(a, fill(nil, size(b)), dims)
        end
    end
end
