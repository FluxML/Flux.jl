module FluxOneAPI

using Flux
using oneAPI
using Adapt
using Zygote
using Zygote: @adjoint

### onehot

using Flux: OneHotArray, OneHotLike

Base.BroadcastStyle(::Type{<:OneHotArray{<:Any, <:Any, <:Any, N, <:oneArray}}) where N =
    oneAPI.oneArrayStyle{N}()

## zygote

# TODO: generalize to GPUArray in Zygote.jl?

@eval @adjoint function Base.broadcasted(::oneAPI.oneArrayStyle, f, args...)
    y, back = Zygote.broadcast_forward(f, args...)
    y, ȳ -> (nothing, nothing, back(ȳ)...)
end

@adjoint oneArray{N,T}(xs::Array) where {N,T} =
    oneArray{N,T}(xs), Δ -> (convert(Array, Δ), )

@adjoint function sum(xs::oneArray; dims = :)
    placeholder = similar(xs)
    sum(xs, dims = dims), Δ -> (placeholder .= Δ,)
end

@adjoint function Base.convert(::Type{T}, xs::Array)  where {T<:oneArray}
    Base.convert(T, xs), Δ -> (nothing, Base.convert(Array, Δ),)
end

function __init__()
    if Flux.default_gpu_converter[] === identity
        @info "Registering oneAPI.jl as the default GPU converter"
        Flux.default_gpu_converter[] = (x)->adapt(oneArray, x)
    else
        @warn "Not registering oneAPI.jl as the default GPU converter as another one has been registered already."
    end
end

end # module
