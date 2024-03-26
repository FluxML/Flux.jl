# v0.15 deprecations

# Enable these when 0.15 is released, and delete const ClipGrad = Optimise.ClipValue etc: 
# Base.@deprecate_binding Optimiser OptimiserChain
# Base.@deprecate_binding ClipValue ClipGrad

train!(loss::Function, ps::Zygote.Params, data, opt) = throw(ArgumentError(
  """On Flux 0.15, `train!` no longer accepts implicit `Zygote.Params`.
  Instead of `train!(loss_xy, Flux.params(model), data, Adam())`
  it now needs `opt = Flux.setup(Adam(), model); train!(loss_mxy, model, data, opt)`
  where `loss_mxy` accepts the model as its first argument.
  """
))
