function (c::Conv)(x::T) where T <: ROCArray
    Flux._size_check(c, x, ndims(x) - 1 => Flux._channels_in(c))
    σ = NNlib.fast_act(c.σ, x)
    cdims = DenseConvDims(
        x, c.weight; stride=c.stride, padding=c.pad,
        dilation=c.dilation, groups=c.groups, flipkernel=true)
    xT = Flux._match_eltype(c, x)
    σ.(conv(xT, c.weight, cdims) .+ conv_reshape_bias(c))
end
