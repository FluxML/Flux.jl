export Affine

# TODO: type hints for parameters

@net type Affine
  W
  b
  x -> x*W .+ b
end

Affine(in::Integer, out::Integer; init = initn) =
  Affine(init(in, out), init(1, out))

@net type Sigmoid
  layer::Model
  x -> σ(layer(x))
end

Sigmoid(in::Integer, out::Integer; init = randn) =
  Sigmoid(Affine(in, out, init = init))
