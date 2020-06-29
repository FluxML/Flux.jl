module Flux

# Zero Flux Given

using Base: tail
using Statistics, Random, LinearAlgebra
using Zygote, MacroTools, Juno, Reexport
using MacroTools: @forward
@reexport using NNlib
using Zygote: Params, @adjoint, gradient, pullback, @nograd

export gradient

export Chain, Dense, Maxout, RNN, LSTM, GRU, SamePad, Conv, CrossCor, ConvTranspose,
       GlobalMaxPool, GlobalMeanPool, MaxPool, MeanPool, flatten,
       DepthwiseConv, Dropout, AlphaDropout, LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
       GaussianNoise, SkipConnection, params, fmap, cpu, gpu, f32, f64, testmode!, trainmode!

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs
export Descent, ADAM, Momentum, Nesterov, RMSProp,
  ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM,
  ADAMW, RADAM, InvDecay, ExpDecay, WeightDecay,
  ClipValue, ClipNorm


using CuArrays
const use_cuda = Ref(false)

include("utils.jl")
include("zeros.jl")
include("onehot.jl")
include("functor.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")

include("data/Data.jl")

include("deprecations.jl")

include("cuda/cuda.jl")

function __init__()
  use_cuda[] = CuArrays.functional() # Can be overridden after load with `Flux.use_cuda[] = false`
  if CuArrays.functional()
    if !CuArrays.has_cudnn()
      @warn "CuArrays.jl found cuda, but did not find libcudnn. Some functionality will not be available."
    end
  end
end

end # module
