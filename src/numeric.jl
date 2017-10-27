using Base.Math: @horner, significand_bits, exponent_raw_max, exponent_bias

if VERSION < v"0.7.0-DEV.1430"
    using Base.Math.fpinttype
else
    using Base.uinttype
end

# log_fast from
# https://github.com/musm/SLEEF.jl/blob/c9dcd2eb090d69ec40790f19798c5fef2aba2616/src/log.jl

const MLN2  = 6.931471805599453094172321214581765680755001343602552541206800094933936219696955e-01 # log(2)

@inline float2integer(d::Float64) = (reinterpret(Int64, d) >> significand_bits(Float64)) % Int
@inline float2integer(d::Float32) = (reinterpret(Int32, d) >> significand_bits(Float32)) % Int

@inline function ilogb2k(d::T) where {T<:Union{Float32,Float64}}
    (float2integer(d) & exponent_raw_max(T)) - exponent_bias(T)
end

@inline function ldexp3k(x::T, e::Int) where {T<:Union{Float32,Float64}}
    if VERSION < v"0.7.0-DEV.1430"
        reinterpret(T, reinterpret(Unsigned, x) + (Int64(e) << significand_bits(T)) % fpinttype(T))
    else
        reinterpret(T, reinterpret(Unsigned, x) + (Int64(e) << significand_bits(T)) % uinttype(T))
    end
end

"""
    log_fast(x)
Compute the natural logarithm of `x`. The inverse of the natural logarithm is
the natural expoenential function `exp(x)`
"""
function log_fast end

let
global log_fast

c8d = 0.153487338491425068243146
c7d = 0.152519917006351951593857
c6d = 0.181863266251982985677316
c5d = 0.222221366518767365905163
c4d = 0.285714294746548025383248
c3d = 0.399999999950799600689777
c2d = 0.6666666666667778740063
c1d = 2.0

c5f = 0.2392828464508056640625f0
c4f = 0.28518211841583251953125f0
c3f = 0.400005877017974853515625f0
c2f = 0.666666686534881591796875f0
c1f = 2f0

global @inline log_fast_kernel(x::Float64) = @horner x c1d c2d c3d c4d c5d c6d c7d c8d
global @inline log_fast_kernel(x::Float32) = @horner x c1f c2f c3f c4f c5f

function log_fast(d::T) where {T<:Union{Float32,Float64}}
    o = d < realmin(T)
    o && (d *= T(Int64(1) << 32) * T(Int64(1) << 32))

    e = ilogb2k(d * T(1.0/0.75))
    m = ldexp3k(d, -e)
    o && (e -= 64)

    x  = (m - 1) / (m + 1)
    x2 = x * x

    t = log_fast_kernel(x2)

    x = x * t + T(MLN2) * e

    isinf(d) && (x = T(Inf))
    (d < 0 || isnan(d)) && (x = T(NaN))
    d == 0 && (x = -T(Inf))

    return x
end
end

log_fast(x::Union{Int32,Int64}) = log_fast(float(x))
