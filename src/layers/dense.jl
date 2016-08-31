export Dense

# TODO: type hints for parameters

@model type Dense
  W
  b
  x -> W*x + b
end

Dense(in::Integer, out::Integer; init = initn) =
  Dense(init(out, in), init(out))

Base.show(io::IO, d::Dense) =
  print(io, "Dense($(size(d.W.x,2)),$(size(d.W.x,1)))")

@model type Sigmoid
  layer::Model
  x -> σ(layer(x))
end

Sigmoid(in::Integer, out::Integer; init = randn) =
  Sigmoid(Dense(in, out, init = init))
