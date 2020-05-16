"""
    BilinearUpsample2d(factors::Tuple{Integer,Integer})

Create an upsampling layer that uses bilinear interpolation to upsample the 1st and 2nd dimension of
a 4-dimensional input array . The size of the output array will be equal to
`(factors[1]*S1, factors[2]*S2, S3, S4)`, where `S1,S2,S3,S4 = size(input_array)`.

# Examples
```jldoctest; setup = :(using Flux: BilinearUpsample2d; using Random; Random.seed!(0))
julia> b = Flux.BilinearUpsample2d((2, 2))
BilinearUpsample2d(2, 2)
julia> b(rand(2, 2, 1, 1))
4×4×1×1 Array{Float64,4}:
[:, :, 1, 1] =
 0.823648  0.658877  0.329336  0.164566
 0.845325  0.675933  0.337149  0.167757
 0.888679  0.710044  0.352773  0.174138
 0.910357  0.7271    0.360586  0.177329```
"""

struct BilinearUpsample2d{T<:Integer}
    factors::Tuple{T,T}
end

BilinearUpsample2d(factor::F) where F<:Integer = BilinearUpsample2d((factor, factor))

@functor BilinearUpsample2d

function (c::T where T<:BilinearUpsample2d)(x::AbstractArray)
    bilinear_upsample2d(x, c.factors)
end

function Base.show(io::IO, l::BilinearUpsample2d)
  print(io, "BilinearUpsample2d( $(l.factors[1]), $(l.factors[2]) )")
end

@adjoint function (c::T where T<:BilinearUpsample2d)(x::AbstractArray)
    (c::T where T<:BilinearUpsample2d)(x), c̄ -> (nothing, bilinear_upsample_adjoint(c̄, c.factors))
end

"""
    `construct_xq(n::T, m::T) where T<:Integer`

Creates interpolation points for resampling, creates the same grid as used in Image.jl `imresize`.
"""
@nograd function construct_xq(n::T, m::T) where T<:Integer
    typed1 = one(n)
    typed2 = 2typed1
    step = n // m
    offset = (n + typed1)//typed2 - step//typed2 - step*(m//typed2 - typed1)
    x = range(offset, step=step, length=m)
    xq = clamp.(x, typed1//typed1, n//typed1)
    return xq
end

"""
    `get_inds_and_ws(xq, dim, n_dims)`

Creates interpolation lower and upper indices, and broadcastable weights
"""
@nograd function get_inds_and_ws(xq, dim, n_dims)
    n = length(xq)

    ilow = floor.(Int, xq)
    ihigh = ceil.(Int, xq)

    wdiff = xq .- ilow

    newsizetup = tuple((i == dim ? n : 1 for i in 1:n_dims)...)
    wdiff = reshape(wdiff, newsizetup)

    return ilow, ihigh, wdiff
end

"""
    adjoint_of_idx(idx ::Vector{T}) where T<:Integer

# Arguments
- `idx::Vector{T<:Integer}`: a vector of indices from which you want the adjoint.

# Outputs
-`idx_adjoint`: index that inverses the operation `x[idx]`.

# Explanation
Determines the adjoint of the vector of indices `idx`, based on the following assumptions:
* `idx[1] == 1`
* `all(d in [0,1] for d in diff(idx))`

The adjoint of `idx` can be seen as an inverse operation such that:
```
x = [1, 2, 3, 4, 5]
idx = [1, 2, 2, 3, 4, 4, 5]
idx_adjoint = adjoint_of_idx(idx)
@assert x[idx][idx_adjoint] == x
```

The above holds as long as `idx` contains every index in `x`.
 """
@nograd function adjoint_of_idx(idx::Vector{T}) where T<:Integer
    d = trues(size(idx))
    d[2:end] .= diff(idx, dims=1)
    idx_adjoint = findall(d)
    return idx_adjoint
end

@nograd function get_newsize(oldsize, k_upsample)
    newsize = (i <= length(k_upsample) ? s*k_upsample[i] : s for (i,s) in enumerate(oldsize))
    return tuple(newsize...)
end

"""
    `bilinear_upsample2d(img::AbstractArray{T,4}, k_upsample::NTuple{2,<:Real}) where T`

# Arguments
- `img::AbstractArray`: the array to be upsampled, must have at least 2 dimensions.
- `k_upsample::NTuple{2}`: a tuple containing the factors with which the first two dimensions of `img` are upsampled.

# Outputs
- `imgupsampled::AbstractArray`: the upsampled version of `img`. The size of `imgupsampled` is
equal to `(k_upsample[1]*S1, k_upsample[2]*S2, S3, S4)`, where `S1,S2,S3,S4 = size(img)`.

# Explanation
Upsamples the first two dimensions of the 4-dimensional array `img` by the two upsample factors stored in `k_upsample`,
using bilinear interpolation. The interpolation grid is identical to the one used by `imresize` from `Images.jl`.
"""
function bilinear_upsample2d(img::AbstractArray{T,4}, k_upsample::NTuple{2,<:Real}) where T

    ilow1, ihigh1, wdiff1, ilow2, ihigh2, wdiff2, ihigh2_r = setup_upsample(size(img), eltype(img), k_upsample)

    @inbounds imgupsampled = bilinear_upsample_workhorse(img, ilow1, ihigh1, wdiff1, ilow2, ihigh2_r, wdiff2)

    return imgupsampled
end

"""
    `bilinear_upsample_workhorse(img, ilowx, ihighx, wdiffx, ilowy, ihigh2_r, wdiffy)`

Does the heavy lifting part of the bilinear upsampling operation
"""
function bilinear_upsample_workhorse(img, ilow1, ihigh1, wdiff1, ilow2, ihigh2_r, wdiff2)
    if typeof(img) <: CuArray
        wdiff1 = CuArray(wdiff1)
        wdiff2 = CuArray(wdiff2)
    end
    imgupsampled = @view(img[ilow1,ilow2,:,:]) .* (1 .- wdiff1) .+ @view(img[ihigh1,ilow2,:,:]) .* wdiff1
    imgupsampled = imgupsampled .* (1 .- wdiff2) .+ @view(imgupsampled[:,ihigh2_r,:,:]) .* wdiff2
end

"""
    `setup_upsample(imgsize::NTuple{4,<:Integer}, imgdtype, k_upsample::NTuple{2,<:Real})`

Creates arrays of interpolation indices and weights for the bilinear_upsample2d operation.
"""
@nograd function setup_upsample(imgsize::NTuple{4,<:Integer}, imgdtype, k_upsample::NTuple{2,<:Real})
    n_dims = 4
    newsize = get_newsize(imgsize, k_upsample)

    # Create interpolation grids
    xq1 = construct_xq(imgsize[1], newsize[1])
    xq2 = construct_xq(imgsize[2], newsize[2])

    # Get linear interpolation lower- and upper index, and weights
    ilow1, ihigh1, wdiff1 = get_inds_and_ws(xq1, 1, n_dims)
    ilow2, ihigh2, wdiff2 = get_inds_and_ws(xq2, 2, n_dims)

    # Adjust the upper interpolation indices of the second dimension
    ihigh2_r = adjoint_of_idx(ilow2)[ihigh2]

    wdiff1 = imgdtype.(wdiff1)
    wdiff2 = imgdtype.(wdiff2)

    return ilow1, ihigh1, wdiff1, ilow2, ihigh2, wdiff2, ihigh2_r

end

"""
 `get_downsamplekernel(n::T) where T<:Integer`

# Arguments
- `n<:Integer`: upsample factor for which a downsample kernel will be determined

# Outputs
- `kernel`: downsample kernel

"""
function get_downsamplekernel(n::T) where T<:Integer
    step = 1//n
    if n % 2 == 0
        start = step//2
        upward = collect(start:step:1//1)
        kernel = [upward; reverse(upward)]
    else
        start = step
        upward = collect(start:step:1//1)
        kernel = [upward; reverse(upward[1:end-1])]
    end
    return kernel
end

"""
    `bilinear_upsample_adjoint(arr::AbstractArray, factors::Tuple{T,T} where T<:Integer)`

# Arguments
- `arr::AbstractArray`: array that has been upsampled using the upsample factors in `factors`

# Outputs
- `arr_ds`: downsampled version of `arr`

# Explanation
Custom adjoint for `BilinearUpsample2d`. Needed because Zygote cannot properly determine gradients
for the current implementation of the forward pass. The adjoint of upsampling is a downsampling operation, which
in this implementation is performed using `Flux.Conv` in combination with a downsampling kernel based on the
upsampling factors. Because of the zero-padding during convolution, the values at the boundary are polluted by edge-effects,
which have been corrected for manually.
"""
function bilinear_upsample_adjoint(arr::AbstractArray, factors::Tuple{T,T} where T<:Integer)

    if size(arr,1) == factors[1]
        arr = sum(arr, dims=1)
        factors = (1, factors[2])
    end

    if size(arr,2) == factors[2]
        arr = sum(arr, dims=2)
        factors = (factors[1], 1)
    end

    if size(arr)[1:2] == (1,1)
        ds_arr = arr
        return ds_arr
    end

    n_chan, n_batch = size(arr)[3:4]

    kern1 = get_downsamplekernel(factors[1])
    kern2 = get_downsamplekernel(factors[2])
    kern = kern1 .* kern2'

    kern_sizes = size(kern)
    pads = tuple((Int.(floor(factor//2)) for factor in factors)...)
    strides = factors

    conv_ds = Conv(kern_sizes, n_chan=>n_chan, pad=pads, stride=strides)

    conv_ds.weight .*= 0
    for i in 1:n_chan
        conv_ds.weight[:,:,i,i] .= kern
    end
    conv_ds.bias .*= 0

    if arr isa CuArray
        conv_ds = gpu(conv_ds)
    end

    arr_ds = conv_ds(arr)

    # Still have to fix edge effects due to zero-padding of convolution,
    # TODO: Could be circumvented by having padding that just extrapolates the value at the first/last index
    nextras = tuple((Int.(floor(factor//2)) for factor in factors)...)

    # First dimension edge-effect correction
    if nextras[1] > 0
        kern_extra1 = kern[1:nextras[1],:]
        conv_extra1 = Conv(size(kern_extra1), n_chan=>n_chan, pad=(0,pads[2]), stride=(1,strides[2]))

        conv_extra1.weight .*= 0
        for i in 1:n_chan
            conv_extra1.weight[:,:,i,i] .= kern_extra1
        end
        conv_extra1.bias .*= 0

        if arr isa CuArray
            conv_extra1 = gpu(conv_extra1)
        end

        arr_ds[[1],:,:,:] .+= conv_extra1(arr[1:nextras[1],:,:,:])
        conv_extra1.weight .= conv_extra1.weight[end:-1:1,:,:,:]
        arr_ds[[end],:,:,:] .+= conv_extra1(arr[end-nextras[1]+1:end,:,:,:])
    end

    # Second dimension edge-effect correction
    if nextras[2] > 0
        kern_extra2 = kern[:,1:nextras[2]]
        conv_extra2 = Conv(size(kern_extra2), n_chan=>n_chan, pad=(pads[1],0), stride=(strides[1],1))

        conv_extra2.weight .*= 0
        for i in 1:n_chan
            conv_extra2.weight[:,:,i,i] .= kern_extra2
        end
        conv_extra2.bias .*= 0

        if arr isa CuArray
            conv_extra2 = gpu(conv_extra2)
        end

        arr_ds[:,[1],:,:] .+= conv_extra2(arr[:,1:nextras[2],:,:])
        conv_extra2.weight .= conv_extra2.weight[:,end:-1:1,:,:]
        arr_ds[:,[end],:,:] .+= conv_extra2(arr[:,end-nextras[2]+1:end,:,:])
    end

    # Finally fix four corners if needed
    kern = eltype(arr).(kern)
    if arr isa CuArray
        kern = gpu(kern)
    end
    n1, n2 = nextras
    if (n1 > 0) & (n2 > 0)
        arr_ds[1,1,:,:] .+= sum(kern[1:n1,1:n2] .* arr[1:n1,1:n2,:,:], dims=(1,2))[1,1,:,:]
        arr_ds[1,end,:,:] .+= sum(kern[1:n1,end-n2+1:end] .* arr[1:n1,end-n2+1:end,:,:], dims=(1,2))[1,1,:,:]
        arr_ds[end,end,:,:] .+= sum(kern[end-n1+1:end,end-n2+1:end] .* arr[end-n1+1:end,end-n2+1:end,:,:], dims=(1,2))[1,1,:,:]
        arr_ds[end,1,:,:] .+= sum(kern[end-n1+1:end,1:n2] .* arr[end-n1+1:end,1:n2,:,:], dims=(1,2))[1,1,:,:]
    end

    return arr_ds
end
