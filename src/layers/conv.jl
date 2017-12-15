struct Conv2D{F,A}
  σ::F
  weight::A
  stride::Int
end

Conv2D(k::NTuple{2,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
       init = initn, stride = 1) =
  Conv2D(σ, param(initn(k..., ch...)), stride)

Flux.treelike(Conv2D)

# (c::Conv2D)(x) = c.σ.(conv2d(x, c.weight, stride = c.stride))
(c::Conv2D)(x) = c.σ.(conv2d(x, c.weight))
