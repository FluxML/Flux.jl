export Dense

@model type Dense
  W
  b
  x -> W*x + b
end

Dense(in::Integer, out::Integer; init = randn) =
  Dense(init(out, in), init(out))
