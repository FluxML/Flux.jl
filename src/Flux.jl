module Flux

# Zero Flux Given

using Base: tail
using Zygote, MacroTools, Juno, Reexport, Statistics, Random
using MacroTools: @forward
@reexport using NNlib
using Zygote: Params, @adjoint, gradient, forward
export gradient

export Chain, Dense, Maxout, RNN, LSTM, GRU, Conv, CrossCor, ConvTranspose, MaxPool, MeanPool,
       DepthwiseConv, Dropout, AlphaDropout, LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
       SkipConnection, params, mapleaves, cpu, gpu, f32, f64

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs
export SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
  ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM,
  ADAMW, RADAM, InvDecay, ExpDecay, WeightDecay

using CUDAapi
if has_cuda()
  try
    using CuArrays
    @eval has_cuarrays() = true
  catch ex
    @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
    @eval has_cuarrays() = false
  end
else
  has_cuarrays() = false
end

include("utils.jl")
include("onehot.jl")
include("treelike.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")

include("data/Data.jl")

include("deprecations.jl")

if has_cuarrays()
  include("cuda/cuda.jl")
end

end # module
