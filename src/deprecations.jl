# Channel notation: Changed to match Conv, but very softly deprecated!
Dense(in::Integer, out::Integer, σ = identity; kw...) =
Dense(in => out, σ; kw...)

Bilinear(in1::Integer, in2::Integer, out::Integer, σ = identity; kw...) =
  Bilinear((in1, in2) => out, σ; kw...)

Embedding(in::Integer, out::Integer; kw...) = Embedding(in => out; kw...)

RNNCell(in::Integer, out::Integer, σ = tanh; kw...) = RNNCell(in => out, σ; kw...)

LSTMCell(in::Integer, out::Integer; kw...) = LSTMCell(in => out; kw...)

GRUCell(in::Integer, out::Integer; kw...) = GRUCell(in => out; kw...)

GRUv3Cell(in::Integer, out::Integer; kw...) = GRUv3Cell(in => out; kw...)

# v0.15 deprecations

Train.train!(loss::Function, ps::Zygote.Params, data, opt) = throw(ArgumentError(
  """On Flux 0.15, `train!` no longer accepts implicit `Zygote.Params`.
  Instead of `train!(loss_xy, Flux.params(model), data, Adam())`
  it now needs `opt_state = Flux.setup(Adam(), model); train!(loss_mxy, model, data, opt_state)`
  where `loss_mxy` accepts the model as its first argument.
  """
))
