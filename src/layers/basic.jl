"""
    Chain(layers...)
    Chain(name = layer, ...)

Collects multiple layers / functions to be called in sequence
on a given input. Supports indexing and slicing, `m[2]` or `m[1:end-1]`,
and if names are given, `m[:name] == m[1]` etc.

# Examples

```jldoctest
julia> m = Chain(x -> x^2, x -> x+1);

julia> m(5) == 26
true

julia> m = Chain(Dense(10, 5, tanh), Dense(5, 2));

julia> x = rand(10, 32);

julia> m(x) == m[2](m[1](x))
true

julia> m2 = Chain(enc = Chain(Flux.flatten, Dense(10, 5, tanh)), 
                  dec = Dense(5, 2));

julia> m2(x) == (m2[:dec] ∘ m2[:enc])(x)
true
```
"""
struct Chain{T}
  layers::T
  Chain(xs...) = new{typeof(xs)}(xs)
  function Chain(; kw...)
    :layers in Base.keys(kw) && throw(ArgumentError("a Chain cannot have a named layer called `layers`"))
    isempty(kw) && return new{Tuple{}}(())
    new{typeof(values(kw))}(values(kw))
  end
end

@forward Chain.layers Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex, Base.keys

functor(::Type{<:Chain}, c) = c.layers, ls -> Chain(ls...)

applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))

(c::Chain)(x) = applychain(Tuple(c.layers), x)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)
Base.getindex(c::Chain{<:NamedTuple}, i::AbstractArray) = 
  Chain(; NamedTuple{Base.keys(c)[i]}(Tuple(c.layers)[i])...)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  _show_layers(io, c.layers)
  print(io, ")")
end
_show_layers(io, layers::Tuple) = join(io, layers, ", ")
_show_layers(io, layers::NamedTuple) = join(io, ["$k = $v" for (k, v) in pairs(layers)], ", ")

# This is a temporary and naive implementation
# it might be replaced in the future for better performance
# see issue https://github.com/FluxML/Flux.jl/issues/702
# Johnny Chen -- @johnnychen94
# only slightly changed to better handle interaction with Zygote @dsweber2
"""
    activations(c::Chain, input)

Calculate the forward results of each layers in Chain `c` with `input` as model input.
"""
activations(c::Chain, input) = extraChain(Tuple(c.layers), input)

function extraChain(fs::Tuple, x)
  res = first(fs)(x)
  return (res, extraChain(Base.tail(fs), res)...)
end
extraChain(::Tuple{}, x) = ()


"""
    Dense(in, out, σ=identity; bias=true, init=glorot_uniform)
    Dense(W::AbstractMatrix, [bias, σ])

Create a traditional `Dense` layer, whose forward pass is given by:

    y = σ.(W * x .+ bias)

The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out`, or a batch with
`size(y) == (out, size(x)[2:end]...)`

Keyword `bias=false` will switch off trainable bias for the layer.
The initialisation of the weight matrix is `W = init(out, in)`, calling the function
given to keyword `init`, with default [`glorot_uniform`](@doc Flux.glorot_uniform).
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.

# Examples
```jldoctest
julia> d = Dense(5, 2)
Dense(5, 2)         # 12 parameters

julia> d(rand(Float32, 5, 64)) |> size
(2, 64)

julia> d(rand(Float32, 5, 1, 1, 64)) |> size  # treated as three batch dimensions
(2, 1, 1, 64)

julia> d1 = Dense(ones(2, 5), false, tanh)  # using provided weight matrix
Dense(5, 2, tanh; bias=false)  # 10 parameters

julia> d1(ones(5))
2-element Vector{Float64}:
 0.9999092042625951
 0.9999092042625951

julia> Flux.params(d1)  # no trainable bias
Params([[1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0]])
```
"""
struct Dense{F, M<:AbstractMatrix, B}
  weight::M
  bias::B
  σ::F
  function Dense(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
    b = create_bias(W, bias, size(W,1))
    new{F,M,typeof(b)}(W, b, σ)
  end
end

function Dense(in::Integer, out::Integer, σ = identity;
               initW = nothing, initb = nothing,
               init = glorot_uniform, bias=true)

  W = if initW !== nothing
    Base.depwarn("keyword initW is deprecated, please use init (which similarly accepts a funtion like randn)", :Dense)
    initW(out, in)
  else
    init(out, in)
  end

  b = if bias === true && initb !== nothing
    Base.depwarn("keyword initb is deprecated, please simply supply the bias vector, bias=initb(out)", :Dense)
    initb(out)
  else
    bias
  end

  return Dense(W, b, σ)
end

@functor Dense

function (a::Dense)(x::AbstractVecOrMat)
  W, b, σ = a.weight, a.bias, a.σ
  return σ.(W*x .+ b)
end

(a::Dense)(x::AbstractArray) = 
  reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.weight, 2), ", ", size(l.weight, 1))
  l.σ == identity || print(io, ", ", l.σ)
  l.bias == Zeros() && print(io, "; bias=false")
  print(io, ")")
end

"""
    Diagonal(α, β)
    Diagonal(size::Integer...)

Create an element-wise linear layer, which performs

    y = α .* x .+ β

The learnable arrays are initialised `α = ones(Float32, size)` and
`β = zeros(Float32, size)`.

Used by [`LayerNorm`](@ref).
"""
struct Diagonal{T}
  α::T
  β::T
end

function Diagonal(sz::Integer...; initα = nothing, initβ = nothing)
  α = if initα !== nothing
    Base.depwarn("keyword initα is deprecated, please simply supply the desired vectors", :Diagonal)
    initα(sz...)
  else
    ones32(sz...)
  end
  β = if initβ !== nothing
    Base.depwarn("keyword initβ is deprecated, please simply supply the desired vectors", :Diagonal)
    initβ(sz...)
  else
    zeros32(sz...)
  end
  Diagonal(α, β)
end

@functor Diagonal

(a::Diagonal)(x) = a.α .* x .+ a.β

function Base.show(io::IO, l::Diagonal)
  print(io, "Diagonal(", join(size(l.α), ", "), ")")
end

"""
    Maxout(layers...)
    Maxout(f, n_alts)

This contains a number of internal layes, each of which receives the same input.
Its output is the elementwise maximum of the the internal layers' outputs.

Instead of defining layers individually, you can provide a zero-argument function
which constructs them, and the number to construct.

Maxout over linear dense layers satisfies the univeral approximation theorem.
See Goodfellow, Warde-Farley, Mirza, Courville & Bengio "Maxout Networks" 
[https://arxiv.org/abs/1302.4389](1302.4389).

See also [`Parallel`](@ref) to reduce with other operators.

# Examples
```
julia> m = Maxout(Dense([1;;], false, abs2), Dense([3;;]));

julia> m([-2 -1 0 1 2])
1×5 Matrix{Int64}:
 4  1  0  3  6

julia> m3 = Maxout(() -> Dense(5, 7, tanh), 3)
Maxout(
  Dense(5, 7, tanh),                    # 42 parameters
  Dense(5, 7, tanh),                    # 42 parameters
  Dense(5, 7, tanh),                    # 42 parameters
)                   # Total: 6 arrays, 126 parameters, 888 bytes.

julia> Flux.outputsize(m3, (5, 11))
(7, 11)
```
"""
struct Maxout{FS<:Tuple}
  over::FS
  Maxout(layers...) = new{typeof(layers)}(layers)
end

function Maxout(f::Function, n_alts::Integer)
  over = Tuple(f() for _ in 1:n_alts)
  return Maxout(over...)
end

@functor Maxout

function (mo::Maxout)(input::AbstractArray)
  # outs = map(lay -> lay(input), mo.over)
  # return max.(outs...)
  # Perhaps surprisingly, pairwise max broadcast is often faster,
  # even with Zygote. See #698 and #1794
  mapreduce(f -> f(input), (acc, out) -> max.(acc, out), mo.over)
end

trainable(mo::Maxout) = mo.over

function Base.show(io::IO, mo::Maxout)
  print(io, "Maxout(")
  _show_layers(io, mo.over)
  print(io, ")")
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

See also [`Parallel`](@ref), [`Maxout`](@ref).
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

"""
    Bilinear(in1, in2, out, σ=identity; bias=true, init=glorot_uniform)
    Bilinear(W::AbstractArray, [bias, σ])

Creates a Bilinear layer, which operates on two inputs at the same time.
Its output, given vectors `x` & `y`, is another vector `z` with,
for all `i ∈ 1:out`:

    z[i] = σ(x' * W[i,:,:] * y + bias[i])

If `x` and `y` are matrices, then each column of the output `z = B(x, y)` is of this form,
with `B` a Bilinear layer.

If `y` is not given, it is taken to be equal to `x`, i.e. `B(x) == B(x, x)`

The two inputs may also be provided as a tuple, `B((x, y)) == B(x, y)`,
which is accepted as the input to a `Chain`.

The initialisation works as for [`Dense`](@ref) layer, with `W = init(out, in1, in2)`.
By default the bias vector is `zeros(Float32, out)`, option `bias=false` will switch off
trainable bias. Either of these may be provided explicitly.

# Examples
```jldoctest
julia> x, y = randn(Float32, 5, 32), randn(Float32, 5, 32);

julia> B = Flux.Bilinear(5, 5, 7);

julia> B(x) |> size  # interactions based on one input
(7, 32)

julia> B(x,y) == B((x,y))  # two inputs, may be given as a tuple
true

julia> sc = SkipConnection(
                Chain(Dense(5, 20, tanh), Dense(20, 9, tanh)),
                Flux.Bilinear(9, 5, 3, bias=false),
            );  # used as the recombinator, with skip as the second input

julia> sc(x) |> size
(3, 32)

julia> Flux.Bilinear(rand(4,8,16), false, tanh)  # first dim of weight is the output
Bilinear(8, 16, 4, tanh, bias=false)
```
"""
struct Bilinear{F,A,B}
  weight::A
  bias::B
  σ::F
  function Bilinear(W::A, bias = true, σ::F = identity) where {A<:AbstractArray, F}
    ndims(A) == 3 || throw(ArgumentError("expected a 3-array of weights"))
    b = create_bias(W, bias, size(W,1))
    new{F,A,typeof(b)}(W, b, σ)
  end
end

@functor Bilinear

function Bilinear(in1::Integer, in2::Integer, out::Integer, σ = identity;
                  init = glorot_uniform, bias = true)
  Bilinear(init(out, in1, in2), bias, σ)
end

function (a::Bilinear)(x::AbstractMatrix, y::AbstractMatrix)
  W, b, σ = a.weight, a.bias, a.σ

  d_z, d_x, d_y = size(W)
  d_x == size(x,1) && d_y == size(y,1) || throw(DimensionMismatch("number of rows in data must match W"))
  size(x,2) == size(y,2) || throw(DimensionMismatch("Data inputs must agree on number of columns, got $(size(x,2)) and $(size(y,2))"))

  # @einsum Wy[o,i,s] := W[o,i,j] * y[j,s]
  Wy = reshape(reshape(W, (:, d_y)) * y, (d_z, d_x, :))

  # @einsum Z[o,s] := Wy[o,i,s] * x[i,s]
  Wyx = batched_mul(Wy, reshape(x, (d_x, 1, :)))
  Z = reshape(Wyx, (d_z, :))

  # @einsum out[o,s] := σ(Z[o,i] + b[o])
  σ.(Z .+ b)
end

(a::Bilinear)(x::AbstractVecOrMat) = a(x, x)
(a::Bilinear)(x::AbstractVector, y::AbstractVector) = vec(a(reshape(x, :,1), reshape(y, :,1)))
(a::Bilinear)(x::NTuple{2, AbstractArray}) = a(x[1], x[2])

function Base.show(io::IO, l::Bilinear)
  print(io, "Bilinear(", size(l.weight, 2), ", ", size(l.weight, 3), ", ", size(l.weight, 1))
  l.σ == identity || print(io, ", ", l.σ)
  l.bias == Flux.Zeros() && print(io, ", bias=false")
  print(io, ")")
end

"""
    Parallel(connection, layers...)
    Parallel(connection; name = layer, ...)

Create a `Parallel` layer that passes an input array to each path in
`layers`, before reducing the output with `connection`.

Called with one input `x`, this is equivalent to `reduce(connection, [l(x) for l in layers])`.
If called with multiple inputs, they are `zip`ped with the layers, thus `Parallel(+, f, g)(x, y) = f(x) + g(y)`.

Like [`Chain`](@ref), its sub-layers may be given names using the keyword constructor.
These can be accessed by indexing: `m[1] == m[:name]` is the first layer.

See also [`SkipConnection`](@ref) which is `Parallel` with one `identity`,
and [`Maxout`](@ref) which reduces by broadcasting `max`.

# Examples

```jldoctest
julia> model = Chain(Dense(3, 5),
                     Parallel(vcat, Dense(5, 4), Chain(Dense(5, 7), Dense(7, 4))),
                     Dense(8, 17));

julia> model(rand(3)) |> size
(17,)

julia> model2 = Parallel(+; α = Dense(10, 2, tanh), β = Dense(5, 2))
Parallel(
  +,
  α = Dense(10, 2, tanh),               # 22 parameters
  β = Dense(5, 2),                      # 12 parameters
)                   # Total: 4 arrays, 34 parameters, 392 bytes.

julia> model2(rand(10), rand(5)) |> size
(2,)

julia> model2[:α](rand(10)) |> size
(2,)

julia> model2[:β] == model2[2]
true
```
"""
struct Parallel{F, T}
  connection::F
  layers::T
end

Parallel(connection, layers...) = Parallel(connection, layers)
function Parallel(connection; kw...)
  layers = NamedTuple(kw)
  if :layers in Base.keys(layers) || :connection in Base.keys(layers)
    throw(ArgumentError("a Parallel layer cannot have a named sub-layer called `connection` or `layers`"))
  elseif isempty(layers)
    Parallel(connection, ())
  end
  Parallel(connection, layers)
end

@functor Parallel

(m::Parallel)(x) = mapreduce(f -> f(x), m.connection, Tuple(m.layers))
(m::Parallel)(xs...) = mapreduce((f, x) -> f(x), m.connection, Tuple(m.layers), xs)
(m::Parallel)(xs::Tuple) = m(xs...)

Base.getindex(m::Parallel, i) = m.layers[i]
Base.getindex(m::Parallel, i::AbstractVector) = Parallel(m.connection, m.layers[i]...)

Base.keys(m::Parallel) = Base.keys(getfield(m, :layers))

trainable(m::Parallel) = (m.connection, m.layers...)

function Base.show(io::IO, m::Parallel)
  print(io, "Parallel(", m.connection, ", ")
  _show_layers(io, m.layers)
  print(io, ")")
end

"""
    Embedding(in, out; init=randn)

A lookup table that stores embeddings of dimension `out` 
for a vocabulary of size `in`. 

This layers is often used to store word embeddings and retrieve them using indices. 
The input to the layer can be either a vector of indexes
or the corresponding [onehot encoding](@ref Flux.OneHotArray). 

# Examples

```julia-repl
julia> using Flux: Embedding

julia> vocab_size, embed_size = 1000, 4;

julia> model = Embedding(vocab_size, embed_size)
Embedding(1000, 4)

julia> vocab_idxs = [1, 722, 53, 220, 3]

julia> x = OneHotMatrix(vocab_idxs, vocab_size);

julia> model(x)
4×5 Matrix{Float32}:
  0.91139    0.670462    0.463217   0.670462    0.110932
  0.247225  -0.0823874   0.698694  -0.0823874   0.945958
 -0.393626  -0.590136   -0.545422  -0.590136    0.77743
 -0.497621   0.87595    -0.870251   0.87595    -0.772696
```

julia> model(vocab_idxs) == model(x)
true
"""
struct Embedding{W}
  weight::W
end

@functor Embedding

Embedding(in::Integer, out::Integer; init = randn32) = Embedding(init(out, in))
  

(m::Embedding)(x::Integer) = m.weight[:, x]
(m::Embedding)(x::AbstractVector) = NNlib.gather(m.weight, x)
(m::Embedding)(x::AbstractArray) = reshape(m(vec(x)), :, size(x)...)

function (m::Embedding)(x::Union{OneHotVector{T,L}, OneHotMatrix{T,L}}) where {T,L}
    size(m.weight, 2) == L || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(m.weight, 2)) != $L"))
  return m(onecold(x))
end
 
function Base.show(io::IO, m::Embedding)
  print(io, "Embedding($(size(m.weight, 2)), $(size(m.weight, 1)))")
end
