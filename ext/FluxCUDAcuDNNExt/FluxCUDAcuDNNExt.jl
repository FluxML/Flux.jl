module FluxCUDAcuDNNExt

using Flux
using CUDA, cuDNN
using NNlib

const USE_CUDNN = Ref{Union{Nothing, Bool}}(nothing)

function check_use_cudnn()
    if !isnothing(USE_CUDNN[])
        return
    end

    USE_CUDNN[] = cuDNN.has_cudnn()
    if !USE_CUDNN[]
        @warn """
        cuDNN.jl didn't found libcudnn, some Flux functionality will not be available.
        """ maxlog=1
    end
    return
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
