export Chain

function inferchain(ms)
  chain = []
  sh = nothing
  for m in ms
    m = init(m, single(sh))
    sh = shape(m, sh)
    push!(chain, m)
  end
  return chain, sh
end

type Chain <: Model
  layers::Vector{Any}
  shape
  function Chain(ms...)
    ms, shape = inferchain(ms)
    return new(ms, shape)
  end
end

@forward Chain.layers Base.getindex, Base.first, Base.last

(s::Chain)(x) = foldl((x, m) -> m(x), x, s.layers)
back!(s::Chain, ∇) = foldr((m, ∇) -> back!(m, ∇), ∇, s.layers)
update!(s::Chain, η) = foreach(l -> update!(l, η), s.layers)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  print_joined(io, c.layers, ", ")
  print(io, ")")
end

graph(s::Chain) =
  foldl((v, m) -> vertex(m, v), constant(ModelInput(1)), s.layers)

shape(c::Chain, in) = c.shape
