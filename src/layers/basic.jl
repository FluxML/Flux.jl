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

julia> m = Chain(Dense(10 => 5, tanh), Dense(5 => 2));

julia> x = rand32(10, 32);

julia> m(x) == m[2](m[1](x))
true

julia> m2 = Chain(enc = Chain(Flux.flatten, Dense(10 => 5, tanh)), 
                  dec = Dense(5 => 2));

julia> m2(x) == (m2[:dec] ∘ m2[:enc])(x)
true
```

For large models, there is a special type-unstable path which can reduce compilation
times. This can be used by supplying a vector of layers `Chain([layer1, layer2, ...])`.
This feature is somewhat experimental, beware!
"""
struct Chain{T<:Union{Tuple, NamedTuple, AbstractVector}}
  layers::T
end

Chain(xs...) = Chain(xs)
function Chain(; kw...)
  :layers in keys(kw) && throw(ArgumentError("a Chain cannot have a named layer called `layers`"))
  isempty(kw) && return Chain(())
  Chain(values(kw))
end

@forward Chain.layers Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex, Base.keys, Base.firstindex

@functor Chain

(c::Chain)(x) = _applychain(c.layers, x)

@generated function _applychain(layers::Tuple{Vararg{Any,N}}, x) where {N}
  symbols = vcat(:x, [gensym() for _ in 1:N])
  calls = [:($(symbols[i+1]) = layers[$i]($(symbols[i]))) for i in 1:N]
  Expr(:block, calls...)
end

_applychain(layers::NamedTuple, x) = _applychain(Tuple(layers), x)

function _applychain(layers::AbstractVector, x)  # type-unstable path, helps compile times
  for f in layers
    x = f(x)
  end
  x
end

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i])
Base.getindex(c::Chain{<:NamedTuple}, i::AbstractArray) =
  Chain(NamedTuple{keys(c)[i]}(Tuple(c.layers)[i]))
function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  _show_layers(io, c.layers)
  print(io, ")")
end

_show_layers(io, layers::Tuple) = join(io, layers, ", ")
_show_layers(io, layers::NamedTuple) = join(io, ["$k = $v" for (k, v) in pairs(layers)], ", ")
_show_layers(io, layers::AbstractVector) = (print(io, "["); join(io, layers, ", "); print(io, "]"))

# This is a temporary and naive implementation
# it might be replaced in the future for better performance
# see issue https://github.com/FluxML/Flux.jl/issues/702
# Johnny Chen -- @johnnychen94
# only slightly changed to better handle interaction with Zygote @dsweber2
"""
    activations(c::Chain, input)

Like calling a `Chain`, but saves the result of each layer as an output.

# Examples

```jldoctest
julia> using Flux: activations

julia> c = Chain(x -> x + 1, x -> x * 2, x -> x ^ 3);

julia> activations(c, 1)
(2, 4, 64)
```
"""
activations(c::Chain, input) = _extraChain(Tuple(c.layers), input)

# Calculates the forward results of each layer provided in a `Tuple` with `x` as model input.
function _extraChain(fs::Tuple, x)
  res = first(fs)(x)
  return (res, _extraChain(Base.tail(fs), res)...)
end
_extraChain(::Tuple{}, x) = ()


"""
    Dense(in => out, σ=identity; bias=true, init=glorot_uniform)
    Dense(W::AbstractMatrix, [bias, σ])

Create a traditional fully connected layer, whose forward pass is given by:

    y = σ.(W * x .+ bias)

The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out`, or a batch with
`size(y) == (out, size(x)[2:end]...)`

Keyword `bias=false` will switch off trainable bias for the layer.
The initialisation of the weight matrix is `W = init(out, in)`, calling the function
given to keyword `init`, with default [`glorot_uniform`](@ref Flux.glorot_uniform).
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.

# Examples
```jldoctest
julia> d = Dense(5 => 2)
Dense(5 => 2)       # 12 parameters

julia> d(rand32(5, 64)) |> size
(2, 64)

julia> d(rand32(5, 6, 4, 64)) |> size  # treated as three batch dimensions
(2, 6, 4, 64)

julia> d1 = Dense(ones(2, 5), false, tanh)  # using provided weight matrix
Dense(5 => 2, tanh; bias=false)  # 10 parameters

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

function Dense((in, out)::Pair{<:Integer, <:Integer}, σ = identity;
               init = glorot_uniform, bias = true)
  Dense(init(out, in), bias, σ)
end

@functor Dense

function (a::Dense)(x::AbstractVecOrMat)
  _size_check(a, x, 1 => size(a.weight, 2))
  σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
  xT = _match_eltype(a, x)  # fixes Float64 input, etc.
  return σ.(a.weight * xT .+ a.bias)
end

function (a::Dense)(x::AbstractArray)
  _size_check(a, x, 1 => size(a.weight, 2))
  reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
end

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.weight, 2), " => ", size(l.weight, 1))
  l.σ == identity || print(io, ", ", l.σ)
  l.bias == false && print(io, "; bias=false")
  print(io, ")")
end

Dense(W::LinearAlgebra.Diagonal, bias = true, σ = identity) =
  Scale(W.diag, bias, σ)

function _size_check(layer, x::AbstractArray, (d, n)::Pair)
  d > 0 || throw(DimensionMismatch(string("layer ", layer,
    " expects ndims(input) > ", ndims(x)-d, ", but got ", summary(x))))
  size(x, d) == n || throw(DimensionMismatch(string("layer ", layer,
    " expects size(input, $d) == $n, but got ", summary(x))))
end
ChainRulesCore.@non_differentiable _size_check(::Any...)

"""
    Scale(size::Integer..., σ=identity; bias=true, init=ones32)
    Scale(scale::AbstractArray, [bias, σ])

Create an element-wise layer, whose forward pass is given by:

    y = σ.(scale .* x .+ bias)

This uses `.*` instead of matrix multiplication `*` of [`Dense`](@ref).
    
The learnable scale & bias are initialised `init(size...)` and `zeros32(size...)`,
with `init=ones32` by default. You may specify the function `init`, 
turn off trainable bias with `bias=false`, or provide the array(s) explicitly.

Used by [`LayerNorm`](@ref) with `affine=true`.

# Examples
```jldoctest
julia> a = Flux.Scale(2)
Scale(2)            # 4 parameters

julia> Flux.params(a)
Params([Float32[1.0, 1.0], Float32[0.0, 0.0]])

julia> a([1 2 3])
2×3 Matrix{Float32}:
 1.0  2.0  3.0
 1.0  2.0  3.0

julia> b = Flux.Scale([1 2 3 4], false, abs2)
Scale(1, 4, abs2; bias=false)  # 4 parameters

julia> b([1, 10])
2×4 Matrix{Int64}:
   1    4    9    16
 100  400  900  1600

julia> Flux.params(b)
Params([[1 2 3 4]])
```
"""
struct Scale{F, A<:AbstractArray, B}
  scale::A
  bias::B
  σ::F
  function Scale(scale::A, bias::B = true, σ::F = identity) where {A<:AbstractArray, B<:Union{Bool, AbstractArray}, F}
    b = create_bias(scale, bias, size(scale)...)
    new{F, A, typeof(b)}(scale, b, σ)
  end
end

Scale(s1::Integer, s23::Integer...; bias = true, init = ones32, _act = identity) = Scale(init(s1, s23...), bias, _act)
Scale(size_act...; bias = true, init = ones32) = Scale(size_act[1:end-1]...; bias, init, _act = size_act[end])

@functor Scale

function (a::Scale)(x::AbstractArray)
  σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
  σ.(a.scale .* x .+ a.bias)
end

function Base.show(io::IO, l::Scale)
  print(io, "Scale(", join(size(l.scale), ", "))
  l.σ == identity || print(io, ", ", l.σ)
  l.bias == false && print(io, "; bias=false")
  print(io, ")")
end

"""
    Maxout(layers...)
    Maxout(f, n_alts)

This contains a number of internal layers, each of which receives the same input.
Its output is the elementwise maximum of the the internal layers' outputs.

Instead of defining layers individually, you can provide a zero-argument function
which constructs them, and the number to construct.

Maxout over linear dense layers satisfies the univeral approximation theorem.
See Goodfellow, Warde-Farley, Mirza, Courville & Bengio "Maxout Networks" 
[https://arxiv.org/abs/1302.4389](https://arxiv.org/abs/1302.4389).

See also [`Parallel`](@ref) to reduce with other operators.

# Examples
```jldoctest
julia> m = Maxout(x -> abs2.(x), x -> x .* 3);

julia> m([-2 -1 0 1 2])
1×5 Matrix{Int64}:
 4  1  0  3  6

julia> m3 = Maxout(() -> Dense(5 => 7, tanh), 3)
Maxout(
  Dense(5 => 7, tanh),                  # 42 parameters
  Dense(5 => 7, tanh),                  # 42 parameters
  Dense(5 => 7, tanh),                  # 42 parameters
)                   # Total: 6 arrays, 126 parameters, 888 bytes.

julia> Flux.outputsize(m3, (5, 11))
(7, 11)
```
"""
struct Maxout{T<:Tuple}
  layers::T
end
Maxout(layers...) = Maxout(layers)
Maxout(f::Function, n_alts::Integer) = Maxout((f() for _ in 1:n_alts)...)

@functor Maxout

function (mo::Maxout)(input::AbstractArray)
  # Perhaps surprisingly, pairwise max broadcast is often faster,
  # even with Zygote. See #698 and #1794
  mapreduce(f -> f(input), (acc, out) -> max.(acc, out), mo.layers)
end

function Base.show(io::IO, mo::Maxout)
  print(io, "Maxout(")
  _show_layers(io, mo.layers)
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
    Bilinear((in1, in2) => out, σ=identity; bias=true, init=glorot_uniform)
    Bilinear(W::AbstractArray, [bias, σ])

Creates a layer which is fully connected between two inputs and the output, and otherwise similar to [`Dense`](@ref).
Its output, given vectors `x` & `y`, is another vector `z` with,
for all `i ∈ 1:out`:

    z[i] = σ(x' * W[i,:,:] * y + bias[i])

If `x` and `y` are matrices, then each column of the output `z = B(x, y)` is of this form,
with `B` the Bilinear layer.

If the second input `y` is not given, it is taken to be equal to `x`, i.e. `B(x) == B(x, x)`

The two inputs may also be provided as a tuple, `B((x, y)) == B(x, y)`,
which is accepted as the input to a `Chain`.

If the two input sizes are the same, `in1 == in2`, then you may write `Bilinear(in => out, σ)`.

The initialisation works as for [`Dense`](@ref) layer, with `W = init(out, in1, in2)`.
By default the bias vector is `zeros(Float32, out)`, option `bias=false` will switch off
trainable bias. Either of these may be provided explicitly.

# Examples
```jldoctest
julia> x, y = randn(Float32, 5, 32), randn(Float32, 5, 32);

julia> B = Flux.Bilinear((5, 5) => 7)
Bilinear(5 => 7)    # 182 parameters

julia> B(x) |> size  # interactions based on one input
(7, 32)

julia> B(x,y) == B((x,y))  # two inputs, may be given as a tuple
true

julia> sc = SkipConnection(
                Chain(Dense(5 => 20, tanh), Dense(20 => 9, tanh)),
                Flux.Bilinear((9, 5) => 3, bias=false),
            );  # used as the recombinator, with skip as the second input

julia> sc(x) |> size
(3, 32)

julia> Flux.Bilinear(rand(4,8,16), false, tanh)  # first dim of weight is the output
Bilinear((8, 16) => 4, tanh; bias=false)  # 512 parameters
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

function Bilinear(((in1, in2), out)::Pair{<:Tuple, <:Integer}, σ = identity;
                  bias = true, init = glorot_uniform)
  Bilinear(init(out, in1, in2), bias, σ)
end
Bilinear((in12, out)::Pair{<:Integer, <:Integer}, σ = identity; kw...) = Bilinear((in12, in12) => out, σ; kw...)

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
  if size(l.weight, 2) == size(l.weight, 3)
    print(io, "Bilinear(", size(l.weight, 2), " => ", size(l.weight, 1))
  else
    print(io, "Bilinear((", size(l.weight, 2), ", ", size(l.weight, 3), ") => ", size(l.weight, 1))
  end
  l.σ == identity || print(io, ", ", l.σ)
  l.bias === false && print(io, "; bias=false")
  print(io, ")")
end

"""
    Parallel(connection, layers...)
    Parallel(connection; name = layer, ...)

Create a layer which passes an input array to each path in
`layers`, before reducing the output with `connection`.

Called with one input `x`, this is equivalent to `connection([l(x) for l in layers]...)`.
If called with multiple inputs, one is passed to each layer, thus `Parallel(+, f, g)(x, y) = f(x) + g(y)`.

Like [`Chain`](@ref), its sub-layers may be given names using the keyword constructor.
These can be accessed by indexing: `m[1] == m[:name]` is the first layer.

See also [`SkipConnection`](@ref) which is `Parallel` with one `identity`,
and [`Maxout`](@ref) which reduces by broadcasting `max`.

# Examples

```jldoctest
julia> model = Chain(Dense(3 => 5),
                     Parallel(vcat, Dense(5 => 4), Chain(Dense(5 => 7), Dense(7 => 4))),
                     Dense(8 => 17));

julia> model(rand32(3)) |> size
(17,)

julia> model2 = Parallel(+; α = Dense(10, 2, tanh), β = Dense(5, 2))
Parallel(
  +,
  α = Dense(10 => 2, tanh),             # 22 parameters
  β = Dense(5 => 2),                    # 12 parameters
)                   # Total: 4 arrays, 34 parameters, 392 bytes.

julia> model2(rand32(10), rand32(5)) |> size
(2,)

julia> model2[:α](rand32(10)) |> size
(2,)

julia> model2[:β] == model2[2]
true
```
"""
struct Parallel{F, T<:Union{Tuple, NamedTuple}}
  connection::F
  layers::T
end

Parallel(connection, layers...) = Parallel(connection, layers)
function Parallel(connection; kw...)
  layers = NamedTuple(kw)
  if :layers in keys(layers) || :connection in keys(layers)
    throw(ArgumentError("a Parallel layer cannot have a named sub-layer called `connection` or `layers`"))
  end
  isempty(layers) && return Parallel(connection, ())
  Parallel(connection, layers)
end

@functor Parallel

(m::Parallel)(x) = m.connection(map(f -> f(x), Tuple(m.layers))...)
(m::Parallel)(xs::Tuple) = m(xs...)

function _parallel_check(layers, xs)
  nl = length(layers)
  nx = length(xs) 
  if (nl != nx)
    throw(ArgumentError("Parallel with $nl sub-layers can take one input or $nl inputs, but got $nx inputs"))
  end
end
ChainRulesCore.@non_differentiable _parallel_check(nl, nx)

function (m::Parallel)(xs...)
  _parallel_check(m.layers, xs)
  m.connection(map(|>, xs, Tuple(m.layers))...)
end

Base.getindex(m::Parallel, i) = m.layers[i]
Base.getindex(m::Parallel, i::AbstractVector) = Parallel(m.connection, m.layers[i])
Base.getindex(m::Parallel{<:Any, <:NamedTuple}, i::AbstractVector) =
  Parallel(m.connection, NamedTuple{keys(m)[i]}(Tuple(m.layers)[i]))

Base.keys(m::Parallel) = keys(getfield(m, :layers))

function Base.show(io::IO, m::Parallel)
  print(io, "Parallel(", m.connection, ", ")
  _show_layers(io, m.layers)
  print(io, ")")
end

"""
    PairwiseFusion(connection, layers...)

## Arguments

- `connection`: A function taking 2 inputs and combining them into a single output 
- `layers`: The layers whose outputs are combined

## Inputs

This layer behaves differently based on input type:

1. If input `x` is a tuple of length N (or the input is `xs` with N `x`'s), matching the number of `layers`, 
  then each layer receives a new input `x[i]` combined with the previous output `y[i-1]` using `connection`.
  Thus `(y1, y2, y3) = PairwiseFusion(connection, layer1, layer2, layer3)((x1, x2, x3))`
  may be drawn as:
```
x1 → layer1 → y1 ↘
                  connection → layer2 → y2 ↘
              x2 ↗                          connection → layer3 → y3
                                        x3 ↗
```
... or written as:
```julia
y1 = layer1(x1)
y2 = layer2(connection(x2, y1))
y3 = layer3(connection(x3, y2))
```

2. With just one input, each layer receives the same `x` combined with the previous output.
   Thus `y = PairwiseFusion(connection, layers...)(x)` obeys:

```julia
y[1] == layers[1](x)
for i in 2:length(layers)
    y[i] == connection(x, layers[i](y[i-1]))
end
```

## Returns

A tuple of length N with the output of each fusion ((`y1`, `y2`, ..., `yN`) in the example above).
"""
struct PairwiseFusion{F, T<:Union{Tuple, NamedTuple}}
  connection::F
  layers::T
end

PairwiseFusion(connection, layers...) = PairwiseFusion(connection, layers)
function PairwiseFusion(connection; kw...)
  layers = NamedTuple(kw)
  if :layers in keys(layers) || :connection in keys(layers)
    throw(ArgumentError("a PairwiseFusion layer cannot have a named sub-layer called `connection` or `layers`"))
  end
  isempty(layers) && return PairwiseFusion(connection, ())
  PairwiseFusion(connection, layers)
end

function _pairwise_check(x, layers, T)
  lx = length(x)
  N = length(layers)
  if T <: Tuple && lx != N
    throw(ArgumentError("PairwiseFusion with $N sub-layers can take one input or $N inputs, but got $lx inputs"))
  end
end
ChainRulesCore.@non_differentiable _pairwise_check(lx, N, T)

function (m::PairwiseFusion)(x::T) where {T}
  _pairwise_check(x, m.layers, T)
  applypairwisefusion(m.layers, m.connection, x)
end
(m::PairwiseFusion)(xs...) = m(xs)

@generated function applypairwisefusion(layers::Tuple{Vararg{Any,N}}, connection, x::T) where {N, T}
  y_symbols = [gensym() for _ in 1:(N + 1)]
  getinput(i) = T <: Tuple ? :(x[$i]) : :x
  calls = [:($(y_symbols[N + 1]) = $(getinput(1)))]
  for i in 1:N - 1
    push!(calls, quote
      $(y_symbols[i]) = layers[$i]($(y_symbols[N + 1]))
      $(y_symbols[N + 1]) = connection($(y_symbols[i]), $(getinput(i + 1)))
    end)
  end
  push!(calls, :($(y_symbols[N]) = layers[$N]($(y_symbols[N + 1]))))
  push!(calls, :(return tuple($(Tuple(y_symbols[1:N])...))))
  return Expr(:block, calls...)
end
applypairwisefusion(layers::NamedTuple, connection, x) = applypairwisefusion(Tuple(layers), connection, x)

@functor PairwiseFusion

Base.getindex(m::PairwiseFusion, i) = m.layers[i]
Base.getindex(m::PairwiseFusion, i::AbstractVector) = PairwiseFusion(m.connection, m.layers[i])
Base.getindex(m::PairwiseFusion{<:Any, <:NamedTuple}, i::AbstractVector) =
  PairwiseFusion(m.connection, NamedTuple{keys(m)[i]}(Tuple(m.layers)[i]))

Base.keys(m::PairwiseFusion) = keys(getfield(m, :layers))

function Base.show(io::IO, m::PairwiseFusion)
  print(io, "PairwiseFusion(", m.connection, ", ")
  _show_layers(io, m.layers)
  print(io, ")")
end

"""
    Embedding(in => out; init=randn32)

A lookup table that stores embeddings of dimension `out` 
for a vocabulary of size `in`, as a trainable matrix.

This layer is often used to store word embeddings and retrieve them using indices. 
The input to the layer can be a vocabulary index in `1:in`, an array of indices,
or the corresponding [`onehot encoding`](@ref OneHotArrays.onehotbatch).

For indices `x`, the result is of size `(out, size(x)...)`, allowing several batch dimensions.
For one-hot `ohx`, the result is of size `(out, size(ohx)[2:end]...)`.

# Examples
```jldoctest
julia> emb = Embedding(26 => 4, init=Flux.identity_init(gain=22))
Embedding(26 => 4)  # 104 parameters

julia> emb(2)  # one column of e.weight (here not random!)
4-element Vector{Float32}:
  0.0
 22.0
  0.0
  0.0

julia> emb([3, 1, 20, 14, 4, 15, 7])  # vocabulary indices, in 1:26
4×7 Matrix{Float32}:
  0.0  22.0  0.0  0.0   0.0  0.0  0.0
  0.0   0.0  0.0  0.0   0.0  0.0  0.0
 22.0   0.0  0.0  0.0   0.0  0.0  0.0
  0.0   0.0  0.0  0.0  22.0  0.0  0.0

julia> ans == emb(Flux.onehotbatch("cat&dog", 'a':'z', 'n'))
true

julia> emb(rand(1:26, (10, 1, 12))) |> size  # three batch dimensions
(4, 10, 1, 12)
```
"""
struct Embedding{W<:AbstractMatrix}
  weight::W
end

@functor Embedding

Embedding((in, out)::Pair{<:Integer, <:Integer}; init = randn32) = Embedding(init(out, in))

(m::Embedding)(x::Integer) = m.weight[:, x]
(m::Embedding)(x::AbstractVector) = NNlib.gather(m.weight, x)
(m::Embedding)(x::AbstractArray) = reshape(m(vec(x)), :, size(x)...)

(m::Embedding)(x::AbstractVector{Bool}) = m.weight * x  # usually OneHotVector
(m::Embedding)(x::AbstractMatrix{Bool}) = m.weight * x  # usually OneHotMatrix
(m::Embedding)(x::AbstractArray{Bool}) = reshape(m(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, m::Embedding)
  print(io, "Embedding(", size(m.weight, 2), " => ", size(m.weight, 1), ")")
end


"""
    _splitat(data::AbstractVector, offsets::AbstractVector{Int})

Splits a vector of data into a vector of vectors based on offsets. Each offset
specifies the next sub-vectors starting index in the `data` vector. In otherwords,
the `data` vector is chuncked into vectors from `offsets[1]` to `offsets[2]` (not including the element at `offsets[2]`), `offsets[2]` to `offsets[3]`, etc.
The last offset specifies a bag that contains everything to the right of it.

The `offsets` vector must begin with `1` and be monotonically increasing. The last element of `offsets` must be at most `length(data)`.
"""
function _splitat(data::AbstractVector, offsets::AbstractVector{Int})
  offsets[firstindex(offsets)] == 1 || throw(ArgumentError("`offsets` must begin with 1."))
  offsets[end] <= length(data) || throw(ArgumentError("The last element in `offsets` must be at most the length of `data`."))
  issorted(offsets, lt = <=) || throw(ArgumentError("`offsets` must be monotonically increasing with no duplicates."))
  newoffsets = vcat(offsets, [lastindex(data)])
  return [data[offsets[i]:(i+1 > lastindex(offsets) ? end : offsets[i+1]-1)] for i in eachindex(offsets)]
end

"""
    EmbeddingBag(in => out, reduction=mean; init=Flux.randn32)

A lookup table that stores embeddings of dimension `out` for a vocabulary of size `in`.
Differs from [`Embedding`](@ref) in that, instead of acting on a single vocabulary index,
it always acts a vector of indices which it calls a "bag".
Their individual embedding vectors are reduced to one, using `mean` or some other function.

Instead of acting on one "bag", such as `x::Vector{Int}`, the layer can also act on several:

* Acting on a vector of "bags", it produces a matrix whose columns are the reduced vectors.
  More generally on `x::Array{Vector{Int}}`, its output is of size `(out, size(x)...)`.

* Any higher-rank array of integers is interpreted as a collection of "bags" each along the first dimension.
  Thus the output is `mapslices(e, x; dims=1)` when `e::EmbeddingBag` and `x::Array{Int,N}`.
  This method is more efficient, but requires that all "bags" have the same length.

* A vector of "bags" may also be produced by splitting a vector of indices at specified points.
  For this case the layer takes two inputs, both vectors of integers. See details below.

The "bag" may equivalently be represented as a `OneHotMatrix`. A collection of these,
or one higher-rank `OneHotArray`, again produce a stack of embeddings. See details below.

# Examples
```jldoctest
julia> vocab_size = 26;  # embed into 3 dimensions, with non-random vectors:

julia> eb = EmbeddingBag(vocab_size => 3, init=Flux.identity_init(gain=100))
EmbeddingBag(26 => 3)  # 78 parameters

julia> eb([2])  # one bag of 1 item
3-element Vector{Float32}:
   0.0
 100.0
   0.0

julia> eb([3,3,1])  # one bag of 3 items, one mean embedding
3-element Vector{Float32}:
 33.333332
  0.0
 66.666664

julia> eb([[3,1,3], [2,1]])  # two bags
3×2 Matrix{Float32}:
 33.3333  50.0
  0.0     50.0
 66.6667   0.0

julia> eb([1 1 1 1; 1 2 3 4])  # 4 bags each of 2 items, eachcol([1 1 1 1; 1 2 3 4])
3×4 Matrix{Float32}:
 100.0  50.0  50.0  50.0
   0.0  50.0   0.0   0.0
   0.0   0.0  50.0   0.0

julia> eb(rand(1:26, 10, 5, 5)) |> size  # 25 bags each of 10 items
(3, 5, 5)
```

Another way to specify "many bags of many items" is to provide a vector `data` (each in `1:in`)
and a vector `at` stating where to split that up into "bags".
The first bag starts with `data[at[1]]`, the second at `data[at[2]]`, and so on, 
with no overlaps and nothing left out (thus it requires `at[1]==1`).

```jldoctest
julia> data = [11, 1, 12, 2, 13, 3, 14];

julia> Flux._splitat(data, [1, 4]) |> println  # internal function, makes data[1:3], data[4:end]
[[11, 1, 12], [2, 13, 3, 14]]

julia> eb(data, [1, 4])  # two bags, of 3 and 4 items
3×2 Matrix{Float32}:
 33.3333   0.0
  0.0     25.0
  0.0     25.0
```

Finally, each bag may also be also be represented as a [`OneHotMatrix`](@ref OneHotArrays.onehotbatch).

```jldoctest
julia> eb(Flux.onehotbatch("bba", 'a':'z'))  # same as [2,2,1], one bag of 3 items
3-element Vector{Float32}:
 33.333332
 66.666664
  0.0

julia> eb([Flux.onehotbatch("bba", 'a':'z'), Flux.onehotbatch("cc", 'a':'z')])  # two bags
3×2 Matrix{Float32}:
 33.3333    0.0
 66.6667    0.0
  0.0     100.0
```
"""
struct EmbeddingBag{F, W<:AbstractMatrix}
  weight::W
  reduction::F
end

@functor EmbeddingBag

EmbeddingBag((in, out)::Pair{<:Integer, <:Integer}, reduction::Function = mean; init = randn32) = EmbeddingBag(init(out, in), reduction)
EmbeddingBag(weight) = EmbeddingBag(weight, mean)

function (m::EmbeddingBag)(data::AbstractVector, offsets::AbstractVector)
  return m(_splitat(data, offsets))
end
(m::EmbeddingBag)(inds::AbstractArray{<:Integer}) = dropdims(m.reduction(Embedding(m.weight)(inds), dims=2), dims=2)
(m::EmbeddingBag)(ind::Integer) = error("EmbeddingBag expects an array of indices, not just one")

(m::EmbeddingBag)(hot::AbstractArray{Bool}) = dropdims(m.reduction(Embedding(m.weight)(hot), dims=2), dims=2)
(m::EmbeddingBag)(hot::AbstractVector{Bool}) = error("EmbeddingBag not defined for a one-hot vector")

# These two could be stack(m, bags), but no AD support yet. (Gradient for weight quite inefficient here.)
(m::EmbeddingBag)(bags::AbstractVector{<:AbstractVector}) = reduce(hcat, m.(bags))
(m::EmbeddingBag)(bags::AbstractArray{<:AbstractVector}) = reshape(m(vec(bags)), :, size(bags)...)

(m::EmbeddingBag)(bags::AbstractArray{<:AbstractMatrix{Bool}}) = reshape(reduce(hcat, m.(vec(bags))), :, size(bags)...)

function Base.show(io::IO, m::EmbeddingBag)
  print(io, "EmbeddingBag(", size(m.weight, 2), " => ", size(m.weight, 1), ")")
end
