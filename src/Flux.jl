module Flux

# Zero Flux Given

using Base: tail
using Statistics, Random, LinearAlgebra
using Zygote, MacroTools, Juno, Reexport
using MacroTools: @forward
@reexport using NNlib
using Zygote: Params, @adjoint, gradient, pullback, @nograd
using GPUArrays

export gradient

export Chain, Dense, Maxout, SkipConnection, Parallel, flatten,
       RNN, LSTM, GRU,
       SamePad, Conv, CrossCor, ConvTranspose, DepthwiseConv,
       AdaptiveMaxPool, AdaptiveMeanPool, GlobalMaxPool, GlobalMeanPool, MaxPool, MeanPool,
       Dropout, AlphaDropout, LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
       Upsample, PixelShuffle,
       params, fmap, cpu, gpu, f32, f64,
       testmode!, trainmode!

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs
using .Optimise: skip
export Descent, ADAM, Momentum, Nesterov, RMSProp,
  ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, OADAM,
  ADAMW, RADAM, AdaBelief, InvDecay, ExpDecay,
  WeightDecay, ClipValue, ClipNorm


const default_gpu_converter = Ref{Function}(identity)

include("utils.jl")
include("zeros.jl")
include("onehot.jl")
include("functor.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")
include("layers/upsample.jl")

include("outputsize.jl")

include("data/Data.jl")

include("losses/Losses.jl")
using .Losses # TODO: stop importing Losses in Flux's namespace in v0.12

include("deprecations.jl")

end # module
