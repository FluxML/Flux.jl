export Sequence

type Sequence <: Model
  layers::Vector{Model}
end

Sequence() = Sequence([])

@forward Sequence.layers Base.getindex, Base.first, Base.last

Base.push!(s::Sequence, m::Model) = push!(s.layers, m)

Base.push!(s::Sequence, f::Init) = push!(s, f(shape(last(s))))

function Sequence(ms...)
  s = Sequence()
  foreach(m -> push!(s, m), ms)
  return s
end

(s::Sequence)(x) = foldl((x, m) -> m(x), x, s.layers)
back!(s::Sequence, ∇) = foldr((m, ∇) -> back!(m, ∇), ∇, s.layers)
update!(s::Sequence, η) = foreach(l -> update!(l, η), s.layers)
