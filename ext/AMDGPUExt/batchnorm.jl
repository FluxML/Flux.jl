function (b::Flux.BatchNorm)(x::ROCArray{T}) where T <: MIOPENFloat
    bλ.(_amd_batchnorm(x, b.γ, b.β; μ=b.μ, σ²=b.σ², ϵ=b.ϵ))
end

function _amd_batchnorm(x, γ, β; μ, σ², ϵ)
    if NNlib.within_gradient(x)
        return AMDGPU.MIOpen.batchnorm_training(x, γ, β, μ, σ²; ϵ, iteration=0) # TODO iteration
    else
        return AMDGPU.MIOpen.batchnorm_inference(x, γ, β, μ, σ²; ϵ)
    end
end

function ChainRulesCore.rrule(::typeof(_amd_batchnorm), x, γ, β; μ, σ², ϵ)
    y, μ_saved, ν_saved = _amd_batchnorm(x, γ, β; μ, σ², ϵ)
    function _batchnorm_pullback(Δ)
        dx, dγ, dβ = MIOpen.∇batchnorm(Δ, x, γ, β, μ_saved, ν_saved)
        (NoTangent(), dx, dγ, dβ)
    end
    y, _batchnorm_pullback
end
