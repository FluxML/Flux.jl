"""
    Chain(layers...)

Chain multiple layers / functions together, so that they are called in sequence
on a given input.

    m = Chain(x -> x^2, x -> x+1)
    m(5) == 26

    m = Chain(Dense(10, 5), Dense(5, 2))
    x = rand(10)
    m(x) == m[2](m[1](x))

`Chain` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1:3](x)` will calculate the output of the first three layers.
"""
type Chain
  layers::Vector{Any}
  Chain(xs...) = new([xs...])
end

@forward Chain.layers Base.getindex, Base.first, Base.last, Base.endof, Base.push!
@forward Chain.layers Base.start, Base.next, Base.done

children(c::Chain) = c.layers
mapchildren(f, c::Chain) = Chain(f.(c.layers)...)

(s::Chain)(x) = foldl((x, m) -> m(x), x, s.layers)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

"""
    Dense(in::Integer, out::Integer, σ = identity)

Creates a traditional `Dense` layer with parameters `W` and `b`.

    y = σ.(W * x .+ b)

The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `in`.
"""
struct Dense{F,S,T}
  σ::F
  W::S
  b::T
end

Dense(in::Integer, out::Integer, σ = identity; init = initn) =
  Dense(σ, param(init(out, in)), param(init(out)))

treelike(Dense)

function (a::Dense)(x)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
