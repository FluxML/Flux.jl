# Chain

type Chain
  layers::Vector{Any}
  Chain(xs...) = new([xs...])
end

@forward Chain.layers Base.getindex, Base.first, Base.last, Base.endof, Base.push!
@forward Chain.layers Base.start, Base.next, Base.done

(s::Chain)(x) = foldl((x, m) -> m(x), x, s.layers)

Compiler.graph(s::Chain) =
  foldl((v, m) -> vertex(m, v), constant(inputnode(1)), s.layers)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

# Affine

struct Affine{S,T}
  W::S
  b::T
end

Affine(in::Integer, out::Integer; init = initn) =
  Affine(track(init(out, in)),
         track(init(out)))

(a::Affine)(x) = a.W*x .+ a.b
