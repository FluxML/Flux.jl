module FluxAMDGPUExt

import ChainRulesCore
import ChainRulesCore: NoTangent, unthunk
import Flux
import Flux: fmap, DenseConvDims, Conv, ConvTranspose, conv, conv_reshape_bias
import NNlib

using MLDataDevices
using AMDGPU
using Adapt
using Random
using Zygote

const MIOPENFloat = AMDGPU.MIOpen.MIOPENFloat

include("functor.jl")
include("batchnorm.jl")
include("conv.jl")

end
