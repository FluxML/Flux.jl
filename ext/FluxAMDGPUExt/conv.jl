function Flux.conv_dims(c::Conv, x::T) where T <: ROCArray
    DenseConvDims(
        x, c.weight; stride=c.stride, padding=c.pad,
        dilation=c.dilation, groups=c.groups, flipkernel=true)
end

function Flux.conv_transpose_dims(c::ConvTranspose, x::T) where T <: ROCArray
    # Calculate combined pad in each dimension
    nd = ndims(x) - 2
    if length(c.pad) == nd
        # Handle symmetric non-constant padding
        combined_pad = ntuple(i -> 2 * c.pad[i], nd)
    else
        combined_pad = ntuple(i -> c.pad[2i-1] + c.pad[2i], nd)
    end

    # Calculate size of "input", from ∇conv_data()'s perspective...
    I = (size(x)[1:end - 2] .- 1) .* c.stride .+ 1 .+
        (size(c.weight)[1:end - 2] .- 1) .* c.dilation .- combined_pad .+ c.outpad
    C_in = size(c.weight)[end - 1] * c.groups
    batch_size = size(x)[end]

    # Create DenseConvDims() that looks like the corresponding conv().
    w_size = size(c.weight)
    DenseConvDims(
        (I..., C_in, batch_size), w_size;
        stride=c.stride, padding=c.pad, dilation=c.dilation,
        groups=c.groups, flipkernel=true)
end
