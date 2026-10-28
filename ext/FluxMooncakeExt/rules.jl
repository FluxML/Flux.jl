# Mooncake's rules for Flux functions, moved here from Mooncake's own `MooncakeFluxExt`
# (chalk-lab/Mooncake.jl#1371). Loaded only once Mooncake no longer ships that extension;
# see `FluxMooncakeExt.jl`.

using Base: IEEEFloat
import Mooncake:
    DefaultCtx, ReverseMode, rrule!!, @is_primitive, CoDual, zero_fcodual, NoRData

@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(Flux.Losses.mse),DenseArray{P},DenseArray{P}
} where {P<:IEEEFloat}

function rrule!!(
    ::CoDual{typeof(Flux.Losses.mse)}, X::CoDual{<:Array{P}}, Y::CoDual{<:Array{P}}
) where {P<:IEEEFloat}
    function flux_mse_pullback(dloss::P)
        scale = 2dloss / length(X.x)
        @inbounds for n in eachindex(X.x, Y.x, X.dx, Y.dx)
            d = X.x[n] - Y.x[n]
            X.dx[n] += scale * d
            Y.dx[n] -= scale * d
        end
        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(Flux.Losses.mse(X.x, Y.x)), flux_mse_pullback
end

function rrule!!(
    ::CoDual{typeof(Flux.Losses.mse)},
    X::CoDual{<:DenseArray{P}},
    Y::CoDual{<:DenseArray{P}},
) where {P<:IEEEFloat}
    function flux_mse_pullback(dloss::P)
        scale = 2dloss / length(X.x)
        @. X.dx += scale * (X.x - Y.x)
        @. Y.dx -= scale * (X.x - Y.x)
        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(Flux.Losses.mse(X.x, Y.x)), flux_mse_pullback
end
