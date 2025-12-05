module FluxCUDAcuDNNExt

using Flux
using NNlib
using CUDA

try
    using cuDNN
catch
    @warn """Package cuDNN not found in current path.
    - Run `import Pkg; Pkg.add(\"cuDNN\")` to install the cuDNN package, then restart julia.
    - If cuDNN is not installed, some Flux functionalities will not be available when running on the GPU.
    """
end

function (BN::Flux.BatchNorm)(x::Union{CuArray{T,2},CuArray{T,4},CuArray{T,5}},
                              cache=nothing) where T<:Union{Float32, Float64}
  
  @assert BN.affine "BatchNorm: only affine=true supported on gpu"
  @assert length(BN.β) == size(x, ndims(x)-1) "BatchNorm: input has wrong number of channels"

  return BN.λ.(NNlib.batchnorm(BN.γ, BN.β, x, BN.μ, BN.σ², BN.momentum; 
                  cache=cache, alpha=1, beta=0, eps=BN.ϵ, 
                  track_stats=BN.track_stats,
                  training=Flux._isactive(BN, x)))
end



end
