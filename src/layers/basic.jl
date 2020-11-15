"""
    Chain(layers...)

Chain multiple layers / functions together, so that they are called in sequence
on a given input.

`Chain` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1:3](x)` will calculate the output of the first three layers.

# Examples
```jldoctest
julia> m = Chain(x -> x^2, x -> x+1);

julia> m(5) == 26
true

julia> m = Chain(Dense(10, 5), Dense(5, 2));

julia> x = rand(10);

julia> m(x) == m[2](m[1](x))
true
```
"""
struct Chain{T<:Tuple}
  layers::T
  Chain(xs...) = new{typeof(xs)}(xs)
end

@forward Chain.layers Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex

functor(::Type{<:Chain}, c) = c.layers, ls -> Chain(ls...)

applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))

(c::Chain)(x) = applychain(c.layers, x)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

# This is a temporary and naive implementation
# it might be replaced in the future for better performance
# see issue https://github.com/FluxML/Flux.jl/issues/702
# Johnny Chen -- @johnnychen94
# only slightly changed to better handle interaction with Zygote @dsweber2
"""
    activations(c::Chain, input)

Calculate the forward results of each layers in Chain `c` with `input` as model input.
"""
function activations(c::Chain, input)
    extraChain(c.layers, input)
end

function extraChain(fs::Tuple, x)
    res = first(fs)(x)
    return (res, extraChain(Base.tail(fs), res)...)
end

extraChain(::Tuple{}, x) = ()



"""
    Dense(in, out, σ=identity; initW=glorot_uniform, initb=zeros, bias=true)
    Dense(W, b, σ=identity)

Create a traditional `Dense` layer with in×out weight matrix `W` and 
bias vector  `b` of length `out`. The forward pass is given by:

    y = σ.(W * x .+ b)

The input `x` must be a vector of length `in`, a batch of vectors represented
as an `in × N` matrix, or a higher order tensor where all dimensions
after the first one will be treated as batch dimensions.

The out `y` will be a vector  of length `out` or a batch whose first
dimension is `out` and the remaining dimensions are the same as in the input.

Setting `bias` to `false` will switch the bias  off for the layer.

`initW` and `initb` are callables used to initialize weights and biases respectively,
through the calls `initW(out, in)` and `initb(out)`.

# Examples

```julia-repl
julia> d = Dense(5, 2)
Dense(5, 2)

julia> d(rand(Float32, 5))
2-element Array{Float32,1}:
 -0.16210233
  0.123119034

julia> d = Dense(5, 2; bias=false)
Dense(5, 2)
```
"""
struct Dense{F,S<:AbstractArray,T<:Union{Zeros, AbstractVector}}
  W::S
  b::T
  σ::F
end

Dense(W, b) = Dense(W, b, identity)

function Dense(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = zeros, bias=true)
  return Dense(initW(out, in), create_bias(bias, initb, out), σ)
end

@functor Dense

function (a::Dense)(x::AbstractVecOrMat)
  eltype(a.W) == eltype(x) || _dense_typewarn(a, x)
  W, b, σ = a.W, a.b, a.σ
  # reshape to handle dims > 1 as batch dimensions
  sz = size(x)
  x = reshape(x, sz[1], :) 
  x = σ.(W*x .+ b)
  return reshape(x, :, sz[2:end]...)
end

_dense_typewarn(d, x) = @warn "Element types don't match for layer $d, this will be slow." typeof(d.W) typeof(x) maxlog=1
Zygote.@nograd _dense_typewarn

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    Diagonal(in::Integer)

Create an element-wise linear transformation layer with learnable
vectors `α` and `β`:

    y = α .* x .+ β

The input `x` must be a array where `size(x, 1) == in`.
"""
struct Diagonal{T}
  α::T
  β::T
end

Diagonal(in::Integer; initα = ones, initβ = zeros) =
  Diagonal(initα(in), initβ(in))

@functor Diagonal

function (a::Diagonal)(x)
  α, β = a.α, a.β
  α.*x .+ β
end

function Base.show(io::IO, l::Diagonal)
  print(io, "Diagonal(", length(l.α), ")")
end

"""
    Maxout(over)

The [Maxout](https://arxiv.org/abs/1302.4389) layer has a number of
internal layers which all receive the same input. It returns the elementwise
maximum of the internal layers' outputs.

Maxout over linear dense layers satisfies the univeral approximation theorem.
"""
struct Maxout{FS<:Tuple}
    over::FS
end

"""
    Maxout(f, n_alts)

Construct a Maxout layer over `n_alts` instances of the layer given by `f`.
The function takes no arguments and should return some callable layer.
Conventionally, this is a linear dense layer.

# Examples

This constructs a `Maxout` layer over 4 internal dense linear layers, each
identical in structure (784 inputs, 128 outputs):
```jldoctest
julia> insize = 784;

julia> outsize = 128;

julia> Maxout(()->Dense(insize, outsize), 4);
```
"""
function Maxout(f, n_alts)
  over = Tuple(f() for _ in 1:n_alts)
  return Maxout(over)
end

@functor Maxout

function (mo::Maxout)(input::AbstractArray)
    mapreduce(f -> f(input), (acc, out) -> max.(acc, out), mo.over)
end

"""
    SkipConnection(layer, connection)

Create a skip connection which consists of a layer or `Chain` of consecutive
layers and a shortcut connection linking the block's input to the output
through a user-supplied 2-argument callable. The first argument to the callable
will be propagated through the given `layer` while the second is the unchanged,
"skipped" input.

The simplest "ResNet"-type connection is just `SkipConnection(layer, +)`.
Here is a more complicated example:
```jldoctest
julia> m = Conv((3,3), 4 => 7, pad=(1,1));

julia> x = ones(Float32, 5, 5, 4, 10);

julia> size(m(x)) == (5, 5, 7, 10)
true

julia> sm = SkipConnection(m, (mx, x) -> cat(mx, x, dims=3));

julia> size(sm(x)) == (5, 5, 11, 10)
true
```
"""
struct SkipConnection{T,F}
  layers::T
  connection::F  #user can pass arbitrary connections here, such as (a,b) -> a + b
end

@functor SkipConnection

function (skip::SkipConnection)(input)
  skip.connection(skip.layers(input), input)
end

function Base.show(io::IO, b::SkipConnection)
  print(io, "SkipConnection(", b.layers, ", ", b.connection, ")")
end
