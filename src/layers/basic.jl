"""
    Chain(layers...)

Chain multiple layers / functions together, so that they are called in sequence
on a given input.

```julia
m = Chain(x -> x^2, x -> x+1)
m(5) == 26

m = Chain(Dense(10, 5), Dense(5, 2))
x = rand(10)
m(x) == m[2](m[1](x))
```

`Chain` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1:3](x)` will calculate the output of the first three layers.
"""
struct Chain{T<:Tuple}
  layers::T
  Chain(xs...) = new{typeof(xs)}(xs)
end

@forward Chain.layers Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex

functor(c::Chain) = c.layers, ls -> Chain(ls...)

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
    Dense(in::Integer, out::Integer, σ = identity)

Creates a traditional `Dense` layer with parameters `W` and `b`.

    y = σ.(W * x .+ b)

The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `out`.

```julia
julia> d = Dense(5, 2)
Dense(5, 2)

julia> d(rand(5))
Tracked 2-element Array{Float64,1}:
  0.00257447
  -0.00449443
```
"""
struct Dense{F,S,T}
  W::S
  b::T
  σ::F
end

Dense(W, b) = Dense(W, b, identity)

function Dense(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = zeros)
  return Dense(initW(out, in), initb(out), σ)
end

@functor Dense

function (a::Dense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

# Try to avoid hitting generic matmul in some simple cases
# Base's matmul is so slow that it's worth the extra conversion to hit BLAS
(a::Dense{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::Dense{<:Any,W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

"""
    Diagonal(in::Integer)

Creates an element-wise linear transformation layer with learnable
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

`Maxout` is a neural network layer, which has a number of internal layers,
which all have the same input, and the maxout returns the elementwise maximium
of the internal layers' outputs.

Maxout over linear dense layers satisfies the univeral approximation theorem.

Reference:
Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio.
2013. Maxout networks.
In Proceedings of the 30th International Conference on International Conference on Machine Learning - Volume 28 (ICML'13),
Sanjoy Dasgupta and David McAllester (Eds.), Vol. 28. JMLR.org III-1319-III-1327.
https://arxiv.org/pdf/1302.4389.pdf
"""
struct Maxout{FS<:Tuple}
    over::FS
end

"""
    Maxout(f, n_alts)

Constructs a Maxout layer over `n_alts` instances of  the layer given  by `f`.
The function takes no arguement and should return some callable layer.
Conventionally this is a linear dense layer.

For example the following example which
will construct a `Maxout` layer over 4 internal dense linear layers,
each identical in structure (784 inputs, 128 outputs).
```julia
    insize = 784
    outsize = 128
    Maxout(()->Dense(insize, outsize), 4)
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
    SkipConnection(layers, connection)

Creates a Skip Connection, of a layer or `Chain` of consecutive layers
plus a shortcut connection. The connection function will combine the result of the layers
with the original input, to give the final output.

The simplest 'ResNet'-type connection is just `SkipConnection(layer, +)`,
and requires the output of the layers to be the same shape as the input.
Here is a more complicated example:
```
m = Conv((3,3), 4=>7, pad=(1,1))
x = ones(5,5,4,10);
size(m(x)) == (5, 5, 7, 10)

sm = SkipConnection(m, (mx, x) -> cat(mx, x, dims=3))
size(sm(x)) == (5, 5, 11, 10)
```
"""
struct SkipConnection
  layers
  connection  #user can pass arbitrary connections here, such as (a,b) -> a + b
end

@functor SkipConnection

function (skip::SkipConnection)(input)
  skip.connection(skip.layers(input), input)
end

function Base.show(io::IO, b::SkipConnection)
  print(io, "SkipConnection(", b.layers, ", ", b.connection, ")")
end
