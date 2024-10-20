module FluxAMDGPUExt

import Flux
import Flux: adapt_storage, fmap
import Flux: DenseConvDims, Conv, ConvTranspose, conv, conv_reshape_bias
import NNlib
using MLDataDevices: MLDataDevices
using AMDGPU
using Adapt
using Random
using Zygote

const MIOPENFloat = AMDGPU.MIOpen.MIOPENFloat


include("functor.jl")
include("batchnorm.jl")
include("conv.jl")


# TODO
# fail early if input to the model is not on the device (e.g. on the host)
# otherwise we get very cryptic errors & segfaults at the rocBLAS level

end
