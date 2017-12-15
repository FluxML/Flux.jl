struct Conv2D{F,A}
  σ::F
  weight::A
  stride::Int
end

Conv2D(k::NTuple{2,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
       init = initn, stride = 1) =
  Conv2D(σ, param(initn(k..., ch...)), stride)

Flux.treelike(Conv2D)

(c::Conv2D)(x) = c.σ.(conv2d(x, c.weight, stride = c.stride))

function Base.show(io::IO, l::Conv2D)
  print(io, "Conv2D((", size(l.weight, 1), ", ", size(l.weight, 2), ")")
  print(io, ", ", size(l.weight, 3), "=>", size(l.weight, 4))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
