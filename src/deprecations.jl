# v0.13 deprecations

function Broadcast.broadcasted(f::Recur, args...)
  # This had an explicit @adjoint rule, calling Zygote.∇map(__context__, f, args...), until v0.12
  Base.depwarn("""Broadcasting is not safe to use with RNNs, as it does not guarantee an iteration order.
    Re-writing this as a comprehension would be better.""", :broadcasted)
  map(f, args...)  # map isn't really safe either, but 
end

@deprecate frequencies(xs) group_counts(xs)

struct Zeros
  function Zeros()
    Base.depwarn("Flux.Zeros is no more, has ceased to be, is bereft of life, is an ex-boondoggle... please use bias=false instead", :Zeros)
    false
  end
end
Zeros(args...) = Zeros()  # was used both Dense(10, 2, initb = Zeros) and Dense(rand(2,10), Zeros())

function Optimise.update!(x::AbstractArray, x̄)
  Base.depwarn("`Flux.Optimise.update!(x, x̄)` was not used internally and has been removed. Please write `x .-= x̄` instead.", :update!)
  x .-= x̄
end

function Diagonal(size::Integer...; kw...)
  Base.depwarn("Flux.Diagonal is now Flux.Scale, and also allows an activation function.", :Diagonal)
  Scale(size...; kw...)
end
function Diagonal(size::Tuple; kw...)
  Base.depwarn("Flux.Diagonal is now Flux.Scale, and also allows an activation function.", :Diagonal)
  Scale(size...; kw...)
end

# Deprecate this eventually once saving models w/o structure is no more
function loadparams!(m, xs)
  Base.depwarn("loadparams! will be deprecated eventually. Use loadmodel! instead.", :loadparams!)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end

# Channel notation: Changed to match Conv, but very softly deprecated!
# Perhaps change to @deprecate for v0.14, but there is no plan to remove these.
Dense(in::Integer, out::Integer, σ = identity; kw...) =
  Dense(in => out, σ; kw...)
Bilinear(in1::Integer, in2::Integer, out::Integer, σ = identity; kw...) =
  Bilinear((in1, in2) => out, σ; kw...)
Embedding(in::Integer, out::Integer; kw...) = Embedding(in => out; kw...)

RNNCell(in::Integer, out::Integer, σ = tanh; kw...) = RNNCell(in => out, σ; kw...)
LSTMCell(in::Integer, out::Integer; kw...) = LSTMCell(in => out; kw...)

GRUCell(in::Integer, out::Integer; kw...) = GRUCell(in => out; kw...)
GRUv3Cell(in::Integer, out::Integer; kw...) = GRUv3Cell(in => out; kw...)

# Optimisers with old naming convention
Base.@deprecate_binding ADAM Adam
Base.@deprecate_binding NADAM NAdam
Base.@deprecate_binding ADAMW AdamW
Base.@deprecate_binding RADAM RAdam
Base.@deprecate_binding OADAM OAdam
Base.@deprecate_binding ADAGrad AdaGrad
Base.@deprecate_binding ADADelta AdaDelta

# Remove sub-module Data, while making sure Flux.Data.DataLoader keeps working
Base.@deprecate_binding Data Flux false "Sub-module Flux.Data has been removed. The only thing it contained may be accessed as Flux.DataLoader"

@deprecate rng_from_array() default_rng_value()

function istraining()
  Base.depwarn("Flux.istraining() is deprecated, use NNlib.within_gradient(x) instead", :istraining)
  false
end
ChainRulesCore.rrule(::typeof(istraining)) = true, _ -> (NoTangent(),)

function _isactive(m)
  Base.depwarn("_isactive(m) is deprecated, use _isactive(m,x)", :_isactive, force=true)
  _isactive(m, 1:0)
end

## methods for compatibility with the old train!
include("deprecations_train.jl")

# v0.14 deprecations

# Enable these when 0.14 is released, and delete const ClipGrad = Optimise.ClipValue etc: 
# Base.@deprecate_binding Optimiser OptimiserChain
# Base.@deprecate_binding ClipValue ClipGrad

# train!(loss::Function, ps::Zygote.Params, data, opt) = throw(ArgumentError(
#   """On Flux 0.14, `train!` no longer accepts implicit `Zygote.Params`.
#   Instead of `train!(loss_xy, Flux.params(model), data, Adam())`
#   it now needs `opt = Flux.setup(Adam(), model); train!(loss_mxy, model, data, opt)`
#   where `loss_mxy` accepts the model as its first argument.
#   """
# ))
