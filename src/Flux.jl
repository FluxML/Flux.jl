module Flux

using Base: tail
using Compat: @compat # for @compat public
using Preferences
using LinearAlgebra, Statistics, Random  # standard lib
using MacroTools, Reexport, ProgressLogging, SpecialFunctions
using MacroTools: @forward

@reexport using NNlib
using MLUtils
const stack = MLUtils.stack  # now exported by Base
import Optimisers: Optimisers, trainable, destructure  # before v0.13, Flux owned these functions
using Optimisers: freeze!, thaw!, adjust!, trainables
using Random: default_rng
using Zygote, ChainRulesCore
using Zygote: Params, @adjoint, gradient, pullback
using Zygote.ForwardDiff: value
export gradient

@reexport using MLDataDevices: MLDataDevices, gpu_backend!, supported_gpu_backends, reset_gpu_device!,
                    default_device_rng,
                    gpu_device, cpu_device, xla_device,
                    CPUDevice,
                    CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice,
                    XLADevice,
                    # get_device, # we define get_device here for retrocompatibility
                    get_device_type,
                    DeviceIterator


# Pirate error to catch a common mistake. (Internal function `base` because overloading `update!` is more likely to give ambiguities.)
Optimisers.base(dx::Zygote.Grads) = error("Optimisers.jl cannot be used with Zygote.jl's implicit gradients, `Params` & `Grads`")

export Chain, Dense, Embedding, EmbeddingBag,
       Maxout, SkipConnection, Parallel, PairwiseFusion,
       RNN, LSTM, GRU, GRUv3,
       SamePad, Conv, CrossCor, ConvTranspose, DepthwiseConv,
       AdaptiveMaxPool, AdaptiveMeanPool, GlobalMaxPool, GlobalMeanPool, MaxPool, MeanPool,
       Dropout, AlphaDropout,
       LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
       MultiHeadAttention,
       Upsample, PixelShuffle,
       fmap, cpu, gpu, f32, f64, f16, rand32, randn32, zeros32, ones32,
       testmode!, trainmode!

@compat(public, ( # mark unexported symbols as API, on Julia 1.11
  # modules
  Losses, Train,
  # layers
  Bilinear, Scale,
  # utils
  outputsize, state, create_bias, @layer,
))

include("optimise/Optimise.jl")
using .Optimise
export Descent, Adam, Momentum, Nesterov, RMSProp,
  AdaGrad, AdaMax, AdaDelta, AMSGrad, NAdam, OAdam,
  AdamW, RAdam, AdaBelief, InvDecay, ExpDecay,
  WeightDecay, SignDecay, ClipValue, ClipNorm

export ClipGrad, OptimiserChain  # these are const defined in deprecations, for ClipValue, Optimiser

include("train.jl")
using .Train
using .Train: setup

using Adapt, Functors, OneHotArrays
include("utils.jl")
include("functor.jl")

@compat(public, (
  # from OneHotArrays.jl
  onehot, onehotbatch, onecold,  
  # from Functors.jl
  functor, @functor, KeyPath, haskeypath, getkeypath,
  # from Optimise/Train/Optimisers.jl
  setup, update!, destructure, freeze!, adjust!, params, trainable, trainables
))

# Pirate error to catch a common mistake.
Functors.functor(::Type{<:MLUtils.DataLoader}, x) = error("`DataLoader` does not support Functors.jl, thus functions like `Flux.gpu` will not act on its contents.")

include("layers/show.jl")
include("layers/macro.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")
include("layers/upsample.jl")
include("layers/attention.jl")

include("loading.jl")

include("outputsize.jl")
export @autosize

include("deprecations.jl")

include("losses/Losses.jl")
using .Losses

include("devices.jl")

# Distributed Training
include("distributed/backend.jl")
include("distributed/public_api.jl")
export MPIBackend, NCCLBackend, DistributedUtils

@compat(public, (
  # init
  glorot_uniform,
  glorot_normal,
  kaiming_uniform,
  kaiming_normal,
  truncated_normal,
  orthogonal,
  sparse_init,
  identity_init,

  # Losses
  binary_focal_loss,
  binarycrossentropy,
  crossentropy,
  dice_coeff_loss,
  focal_loss,
  hinge_loss,
  huber_loss,
  kldivergence,
  label_smoothing,
  logitbinarycrossentropy,
  logitcrossentropy,
  mae,
  mse,
  msle,
  poisson_loss,
  siamese_contrastive_loss,
  squared_hinge_loss,
  tversky_loss,
))


end # module
