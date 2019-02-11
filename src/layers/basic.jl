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

children(c::Chain) = c.layers
mapchildren(f, c::Chain) = Chain(f.(c.layers)...)

applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))

(c::Chain)(x) = applychain(c.layers, x)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

activations(c::Chain, x) = accumulate((x, m) -> m(x), c.layers, init = x)

"""
    Dense(in::Integer, out::Integer, σ = identity)

Creates a traditional `Dense` layer with parameters `W` and `b`.

    y = σ.(W * x .+ b)

The input `x` must be an array where `size(x, 1) == in`.
The out `y` is similar to `x` except for the `size(y, 1) == out`.

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
  return Dense(param(initW(out, in)), param(initb(out)), σ)
end

@treelike Dense

function (a::Dense)(x::AbstractVecOrMat)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

function (a::Dense)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    x_matrix = reshape(x, size(x, 1), :)
    linear = reshape(W * x_matrix, :, size(x)[2:end]...)
    σ.(linear .+ b)
end

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

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
  Diagonal(param(initα(in)), param(initβ(in)))

@treelike Diagonal

function (a::Diagonal)(x)
  α, β = a.α, a.β
  α.*x .+ β
end

function Base.show(io::IO, l::Diagonal)
  print(io, "Diagonal(", length(l.α), ")")
end

# Try to avoid hitting generic matmul in some simple cases
# Base's matmul is so slow that it's worth the extra conversion to hit BLAS

for t in [AbstractVecOrMat, AbstractArray]
  @eval begin
    (a::Dense{<:Any,W})(x::$t{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
      invoke(a, Tuple{$t}, x)

    (a::Dense{<:Any,W})(x::$t{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
      a(convert.(T, x))
  end
end
