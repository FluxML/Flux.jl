function (b::Flux.BatchNorm)(x::ROCArray{T}) where T <: MIOPENFloat
    b.λ.(_amd_batchnorm(
        x, b.γ, b.β; μ=b.μ, σ²=b.σ², ϵ=b.ϵ,
        within_grad=NNlib.within_gradient(x)))
end

function _amd_batchnorm(x, γ, β; μ, σ², ϵ, within_grad::Bool)
    if within_grad
        return AMDGPU.MIOpen.batchnorm_training(x, γ, β, μ, σ²; ϵ=Float64(ϵ), iteration=0) # TODO iteration
    else
        return AMDGPU.MIOpen.batchnorm_inference(x, γ, β, μ, σ²; ϵ=Float64(ϵ))
    end
end

function ChainRulesCore.rrule(
    ::typeof(_amd_batchnorm), x, γ, β; μ, σ², ϵ, within_grad::Bool,
)
    y, μ_saved, ν_saved = _amd_batchnorm(x, γ, β; μ, σ², ϵ, within_grad)
    function _batchnorm_pullback(Δ)
        dx, dγ, dβ = AMDGPU.MIOpen.∇batchnorm(Δ, x, γ, β, μ_saved, ν_saved)
        (NoTangent(), dx, dγ, dβ)
    end
    y, _batchnorm_pullback
end
