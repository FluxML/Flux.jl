# Chain

type Chain
  layers::Vector{Any}
  Chain(xs...) = new([xs...])
end

@forward Chain.layers Base.getindex, Base.first, Base.last, Base.endof, Base.push!
@forward Chain.layers Base.start, Base.next, Base.done

Optimise.children(c::Chain) = c.layers

(s::Chain)(x) = foldl((x, m) -> m(x), x, s.layers)

Compiler.graph(s::Chain) =
  foldl((v, m) -> vertex(m, v), constant(inputnode(1)), s.layers)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

# Dense

struct Dense{F,S,T}
  σ::F
  W::S
  b::T
end

Dense(in::Integer, out::Integer, σ = identity; init = initn) =
  Dense(σ, param(init(out, in)), param(init(out)))

Optimise.children(d::Dense) = (d.W, d.b)

(a::Dense)(x) = a.σ.(a.W*x .+ a.b)

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
