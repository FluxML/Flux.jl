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

    ilow1, ihigh1, wdiff1, ilow2, ihigh2_r, wdiff2 = setup_upsample(size(img), eltype(img), k_upsample)

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

    return ilow1, ihigh1, wdiff1, ilow2, ihigh2_r, wdiff2

end
