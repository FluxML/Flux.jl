module Flux

using Base: tail
using Compat: @compat # for @compat public
using Preferences
using LinearAlgebra, Statistics, Random  # standard lib
using MacroTools, Reexport, ProgressLogging, SpecialFunctions
using MacroTools: @forward

@reexport using NNlib
using NNlib: conv, ∇conv_data, depthwiseconv, output_size
using MLUtils
using Adapt, OneHotArrays
using Functors: Functors, fmap, fmapstructure

using Optimisers: Optimisers, destructure, freeze!, thaw!, adjust!, trainables, update!
import Optimisers: trainable
@reexport using Optimisers

using Random: default_rng

using Zygote, ChainRulesCore
using Zygote: @adjoint, pullback
using Zygote.ForwardDiff: value
using EnzymeCore: EnzymeCore

@reexport using MLDataDevices: MLDataDevices, supported_gpu_backends, reset_gpu_device!,
                    default_device_rng,
                    gpu_device, cpu_device, xla_device,
                    CPUDevice,
                    CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice,
                    XLADevice,
                    # get_device, # we define get_device here for retrocompatibility
                    gpu_backend!,
                    get_device_type,
                    DeviceIterator

export Chain, Dense, Embedding, EmbeddingBag,
       Maxout, SkipConnection, Parallel, PairwiseFusion,
       RNNCell, LSTMCell, GRUCell, GRUv3Cell,
       RNN, LSTM, GRU, GRUv3,
       SamePad, Conv, CrossCor, ConvTranspose, DepthwiseConv,
       AdaptiveMaxPool, AdaptiveMeanPool, GlobalMaxPool, GlobalMeanPool, MaxPool, MeanPool,
       Dropout, AlphaDropout,
       LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
       MultiHeadAttention,
       Upsample, PixelShuffle,
       fmap, cpu, gpu, f32, f64, f16, rand32, randn32, zeros32, ones32,
       testmode!, trainmode!

@compat(public, ( # unexported symbols marked as API, on Julia 1.11
  # modules
  Losses, Train,
  # layers
  Bilinear, Scale,
  # utils
  outputsize, state, create_bias, @layer, initialstates,
  # from OneHotArrays.jl
  onehot, onehotbatch, onecold,
  # from Train
  setup, train!,
  # from Optimsers.jl
  destructure, freeze!, thaw!, adjust!, trainables, update!, trainable,
  # from Zygote.jl
  hessian, diaghessian, jacobian, withjacobian, pullback,
  # AD functions
  withgradient,
  # init
  glorot_uniform,
  glorot_normal,
  kaiming_uniform,
  kaiming_normal,
  truncated_normal,
  lecun_normal,
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

include("gradient.jl")
export gradient

include("train.jl")
using .Train
using .Train: setup

include("utils.jl")
include("functor.jl")

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

include("losses/Losses.jl")
using .Losses

include("devices.jl")
export get_device

# Distributed Training
include("distributed/backend.jl")
include("distributed/public_api.jl")
export MPIBackend, NCCLBackend, DistributedUtils

include("deprecations.jl")

end # module
