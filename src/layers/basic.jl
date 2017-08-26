# Chain

mutable struct Chain
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

# Linear

struct Linear{F,S,T}
  σ::F
  W::S
  b::T
end

Linear(in::Integer, out::Integer, σ = identity; init = initn) =
  Linear(σ, track(init(out, in)), track(init(out)))

Optimise.children(d::Linear) = (d.W, d.b)

(a::Linear)(x) = a.σ.(a.W*x .+ a.b)

function Base.show(io::IO, l::Linear)
  print(io, "Linear(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
