module FluxAMDGPUExt

import Flux
import Flux: fmap, DenseConvDims, Conv, ConvTranspose, conv, conv_reshape_bias

using MLDataDevices
using AMDGPU
using Adapt
using Random
using Zygote

# The MIOpen `BatchNorm` fast path was removed: `BatchNorm` now wraps `NNlib.batchnorm`,
# which handles `ROCArray`s via its generic path. Moving the MIOpen fast path into NNlib
# (alongside the cuDNN one) is tracked in FluxML/NNlib.jl#752.

include("functor.jl")
include("conv.jl")

end
