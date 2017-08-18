type Chain
  layers::Vector{Any}
  Chain(xs...) = new([xs...])
end

@forward Chain.layers Base.getindex, Base.first, Base.last, Base.endof, Base.push!
@forward Chain.layers Base.start, Base.next, Base.done

(s::Chain)(x) = foldl((x, m) -> m(x), x, s.layers)
update!(s::Chain, η) = foreach(l -> update!(l, η), s.layers)

function back!(s::Chain, Δ, x)
  crumbs = foldl([x], s.layers[1:end-1]) do crumbs, layer
    push!(crumbs, layer(crumbs[end]))
  end

  foldr(Δ, collect(zip(crumbs, s.layers))) do pack, Δ
    x, layer = pack
    back!(layer, Δ, x)
  end
end

graph(s::Chain) =
  foldl((v, m) -> vertex(m, v), constant(inputnode(1)), s.layers)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)
