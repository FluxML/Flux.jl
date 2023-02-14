struct FluxAMDGPUAdaptor end

adapt_storage(::FluxAMDGPUAdaptor, x) = ROCArray(x)
adapt_storage(::FluxAMDGPUAdaptor, x::Zygote.FillArrays.AbstractFill) =
    ROCArray(collect(x))
adapt_storage(::FluxAMDGPUAdaptor, x::Zygote.OneElement) = ROCArray(collect(x))
adapt_storage(::FluxAMDGPUAdaptor, x::Random.TaskLocalRNG) =
    AMDGPU.rocRAND.default_rng()
adapt_storage(::FluxAMDGPUAdaptor, x::AMDGPU.rocRAND.RNG) = x
adapt_storage(::FluxAMDGPUAdaptor, x::AbstractRNG) = error("""
    Cannot map RNG of type $(typeof(x)) to AMDGPU.
    AMDGPU execution only supports Random.default_rng().""")

# TODO adaptor for Conv

adapt_storage(::FluxCPUAdaptor, x::AMDGPU.rocRAND.RNG) = Random.default_rng()

function ChainRulesCore.rrule(::Type{Array}, x::ROCArray)
    Array(x), dx -> (NoTangent(), ROCArray(unthunk(dx)))
end

function ChainRulesCore.rrule(
    ::typeof(Adapt.adapt_storage), to::FluxCPUAdaptor, x::AMDGPU.AnyROCArray,
)
    adapt_storage(to, x), dx -> (
        NoTangent(), NoTangent(),
        adapt_storage(FluxAMDGPUAdaptor(), unthunk(dx)))
end

function _amd(x)
    check_use_amdgpu()
    use_amdgpu[] ? fmap(x -> Adapt.adapt(FluxAMDGPUAdaptor(), x)) : x
end

function check_use_amdgpu()
    use_amdgpu[] === nothing || return

    use_amdgpu[] = AMDGPU.functional()
    if use_amdgpu[]
        if !AMDGPU.functional(:MIOpen)
            @warn "MIOpen is not functional in AMDGPU.jl, some functionality will not be available."
        end
    else
        @info """
        The AMDGPU function is being called but the AMDGPU is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return
end
ChainRulesCore.@non_differentiable check_use_amdgpu()
