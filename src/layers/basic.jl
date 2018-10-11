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

!!! info
* Instance of `Chain` can be used as a function.
* `Chain` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1:3](x)` will calculate the output of the first three layers.
* use `m(x)` if you only want the output of the last layer, use `Flux.activations(m,x)` if you want outputs of each layer.

See also [`Flux.activations`](@ref)
"""
struct Chain
  layers::Vector{Any}
  Chain(xs...) = new([xs...])
end

@forward Chain.layers Base.getindex, Base.first, Base.last, Base.lastindex, Base.push!
@forward Chain.layers Base.iterate

children(c::Chain) = c.layers
mapchildren(f, c::Chain) = Chain(f.(c.layers)...)
adapt(T, c::Chain) = Chain(map(x -> adapt(T, x), c.layers)...)

(c::Chain)(x) = foldl((x, m) -> m(x), c.layers; init = x)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

"""
    activations(c::Chain, x)

The input `c` must be a `Chain`.

Creates an `Array` that stores activation of each layer.

# Examples
```julia
julia> c = Chain(Dense(10,2,σ),Dense(2,1),softmax)
Chain(Dense(10, 2, NNlib.σ), Dense(2, 1), NNlib.softmax)
julia> Flux.activations(c,randn(10))
3-element Array{Any,1}:
 Flux.Tracker.TrackedReal{Float64}[0.923631 (tracked), 0.0163568 (tracked)]
 Flux.Tracker.TrackedReal{Float64}[-0.709397 (tracked)]
 Flux.Tracker.TrackedReal{Float64}[1.0 (tracked)]
```

See also [`Flux.Chain`](@ref)
"""
activations(c::Chain, x) = accumulate((x, m) -> m(x), c.layers, init = x)

"""
    Dense(in::Integer, out::Integer, σ = identity; initW = glorot_uniform, initb = zeros)

Creates a traditional `Dense` layer with parameters `W` and `b`.

    y = σ.(W * x .+ b)

The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `out`.

Instance of `Dense` layer can be used as a function.

# Arguments
- `in::Integer`: the length of input vector `x`.
- `out::Integer`: the length of output vector `y`.
- `σ`: activation function. (Default: `identity`)
- `initW`: method used for initialization of parameters `W`. (Default: `Flux.glorot_uniform`)
- `initb`: method used for initialization of parameters `b`. (Default: `zeros`)


# Examples
```julia
julia> d = Dense(5, 2)
Dense(5, 2)

julia> d(rand(5))
Tracked 2-element Array{Float64,1}:
  0.00257447
  -0.00449443

julia> d(randn(5,2))
5×2 Array{Float64,2}:
4.59414    1.31019
-3.61826    0.0775891
2.4191    -2.59552
0.449825   7.59122
-1.90254   -0.392393
```

See also: [`Flux.Chain`](@ref), [`Flux.glorot_uniform`](@ref), [`Flux.glorot_normal`](@ref),
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

function (a::Dense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    Diagonal(len::Integer; initα = ones, initβ = zeros)

Creates an element-wise linear transformation layer with learnable
vectors `α` and `β`:

    y = α .* x .+ β

The input `x` must be a scalar or an array where `size(x, 1) == in`. If `x` is a scalar, it will be broadcasted if necessary.

Instance of `Dense` layer can be used as a function.

# Arguments
- `len::Integer`: the length of input and output vector `x`.
- `initα`: method used for initialization of parameters `α`. (Default: `ones`)
- `initβ`: method used for initialization of parameters `β`. (Default: `zeros`)

# Examples
```julia
julia> m = Flux.Diagonal(2)
Diagonal(2)

julia> m([1,2])
Tracked 2-element Array{Float64,1}:
 1.0
 2.0

julia> m([1 2]) # broadcasting
Tracked 2×2 Array{Float64,2}:
1.0  2.0
1.0  2.0

julia> m([1 2;3 4])
Tracked 2×2 Array{Float64,2}:
 1.0  2.0
 3.0  4.0
```

See also: [`Flux.Chain`](@ref)
"""
struct Diagonal{T}
  α::T
  β::T
end

Diagonal(len::Integer; initα = ones, initβ = zeros) =
  Diagonal(param(initα(len)), param(initβ(len)))

@treelike Diagonal

function (a::Diagonal)(x)
  α, β = a.α, a.β
  α.*x .+ β
end

function Base.show(io::IO, l::Diagonal)
  print(io, "Diagonal(", length(l.α), ")")
end
