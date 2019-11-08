module Flux

# Zero Flux Given

using Base: tail
using Zygote, MacroTools, Juno, Reexport, Statistics, Random
using MacroTools: @forward
@reexport using NNlib
using Zygote: Params, @adjoint, gradient, pullback
export gradient

export Chain, Dense, Maxout, RNN, LSTM, GRU, SamePad, Conv, CrossCor, ConvTranspose, MaxPool, MeanPool,
       DepthwiseConv, Dropout, AlphaDropout, LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
       SkipConnection, params, fmap, cpu, gpu, f32, f64

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs
export SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
  ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM,
  ADAMW, RADAM, InvDecay, ExpDecay, WeightDecay


ENV["CUDA_INIT_SILENT"] = true
using CUDAdrv, CuArrays
const use_cuda = Ref(false)

include("utils.jl")
include("onehot.jl")
include("functor.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")

include("data/Data.jl")

include("deprecations.jl")

function __init__()
  if !CUDAdrv.functional()
    @warn "CUDA available, but CUDAdrv.jl failed to load"
  elseif length(devices()) == 0
    @warn "CUDA available, but no GPU detected"
  elseif !CuArrays.functional()
    @warn "CUDA GPU available, but CuArrays.jl failed to load"
  else
    use_cuda[] = true

    # FIXME: this functionality should be conditional at run time by checking `use_cuda`
    #        (or even better, get moved to CuArrays.jl as much as possible)
    if CuArrays.has_cudnn()
      include(joinpath(@__DIR__, "cuda/cuda.jl"))
    else
      @warn "CUDA GPU available, but CuArrays.jl did not find libcudnn. Some functionality will not be available."
    end
  end
end

end # module
