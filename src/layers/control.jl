export Chain, @Chain

type Chain
  layers::Vector{Any}
  Chain(xs...) = new([xs...])
end

@forward Chain.layers Base.getindex, Base.first, Base.last, Base.endof, Base.push!
@forward Chain.layers Base.start, Base.next, Base.done

(s::Chain)(x) = foldl((x, m) -> m(x), x, s.layers)
back!(s::Chain, Δ) = foldr((m, Δ) -> back!(m, Δ), Δ, s.layers)
update!(s::Chain, η) = foreach(l -> update!(l, η), s.layers)

graph(s::Chain) =
  foldl((v, m) -> vertex(m, v), constant(inputnode(1)), s.layers)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

# Chain Macros

inferred(f, in, args...; kws...) = f(args...; kws...)

# `inferchain` allows for overriding inference behaviour for convenience.
# For example, `infer(Affine(10, 20), nothing)` would normally return a shape
# error, but for the interface we just ignore any errors and return (1, 20).
inferchain(f, xs...) = infer(f, xs...)

macro Chain(x, xs...)
  inferconstructor(x) =
    @capture(x, f_(xs__)) ? :(inferred($(esc(f)), (shape,), $(esc.(xs)...))) : esc(x)
  @q let
    shape = nothing
    c = Chain($(esc(x)))
    $([:(shape = inferchain(c.layers[end], shape);
         push!(c, $x)) for x in inferconstructor.(xs)]...)
    c
  end
end
