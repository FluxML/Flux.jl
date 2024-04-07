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


function params!(p::Params, x, seen = IdSet())
  # @depwarn "Implicit use of `params` is deprecated. TODO."

  if x isa AbstractArray{<:Number} && Functors.isleaf(x)
    return push!(p, x)
  elseif x in seen
    nothing
  else
    _check_new_macro(x)  # complains if you used @functor not @layer
    push!(seen, x)
    for child in trainable(x)
      params!(p, child, seen)
    end
  end
end

function params(m...)
  # @depwarn "Implicit use of `params` is deprecated. TODO."
  ps = Params()
  params!(ps, m)
  return ps
end

# Allows caching of the parameters when params is called within gradient() to fix #2040.
# @non_differentiable params(m...)  # https://github.com/FluxML/Flux.jl/pull/2054
# That speeds up implicit use, and silently breaks explicit use. 
# From @macroexpand Zygote.@non_differentiable params(m...) and https://github.com/FluxML/Zygote.jl/pull/1248
Zygote._pullback(::Zygote.Context{true}, ::typeof(params), m...) = params(m), _ -> nothing

