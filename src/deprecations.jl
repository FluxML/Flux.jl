# v0.15 deprecations

Train.train!(loss::Function, ps::Zygote.Params, data, opt) = throw(ArgumentError(
  """On Flux 0.15, `train!` no longer accepts implicit `Zygote.Params`.
  Instead of `train!(loss_xy, Flux.params(model), data, Adam())`
  it now needs `opt_state = Flux.setup(Adam(), model); train!(loss_mxy, model, data, opt_state)`
  where `loss_mxy` accepts the model as its first argument.
  """
))
