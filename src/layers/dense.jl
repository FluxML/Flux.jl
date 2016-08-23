export Dense

@model type Dense
  W
  b
  x -> W*x + b
end

Dense(in::Integer, out::Integer; init = randn) =
  Dense(init(out, in), init(out))

@model type Sigmoid
  layer::Model
  x -> σ(layer(x))
end

Sigmoid(in::Integer, out::Integer; init = randn) =
  Sigmoid(Dense(in, out, init = init))
