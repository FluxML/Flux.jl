export Dense

# TODO: type hints for parameters

@model type Dense
  W
  b
  x -> x*W + b
end

Dense(in::Integer, out::Integer; init = initn) =
  Dense(init(in, out), init(1, out))

Base.show(io::IO, d::Dense) =
  print(io, "Dense($(size(d.W.x,1)),$(size(d.W.x,2)))")

@model type Sigmoid
  layer::Model
  x -> σ(layer(x))
end

Sigmoid(in::Integer, out::Integer; init = randn) =
  Sigmoid(Dense(in, out, init = init))
