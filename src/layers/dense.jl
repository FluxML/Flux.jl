export Dense

@model type Dense
  W
  b
  x -> W*x + b
end

Dense(in::Integer, out::Integer; init = randn) =
  Dense(init(out, in), init(out))

Base.show(io::IO, ::Dense) = print(io, "Flux.Dense(...)")

@model type Sigmoid
  layer::Model
  x -> σ(layer(x))
end

Sigmoid(in::Integer, out::Integer; init = randn) =
  Sigmoid(Dense(in, out, init = init))
