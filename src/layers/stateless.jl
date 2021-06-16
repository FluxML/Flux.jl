"""
    flatten(x::AbstractArray)

Reshape arbitrarly-shaped input into a matrix-shaped output,
preserving the size of the last dimension.

See also [`unsqueeze`](@ref).

# Examples
```jldoctest
julia> rand(3,4,5) |> Flux.flatten |> size
(12, 5)

julia> xs = rand(Float32, 10,10,3,7);

julia> m = Chain(Conv((3,3), 3=>4, pad=1), Flux.flatten, Dense(400,33));

julia> xs |> m[1] |> size
(10, 10, 4, 7)

julia> xs |> m |> size
(33, 7)
```
"""
function flatten(x::AbstractArray)
  return reshape(x, :, size(x)[end])
end

"""
    normalise(x; dims=ndims(x), ϵ=1e-5)

Normalise `x` to mean 0 and standard deviation 1 across the dimension(s) given by `dims`.
Per default, `dims` is the last dimension. 
`ϵ` is a small additive factor added to the denominator for numerical stability.
"""
function normalise(x::AbstractArray; dims=ndims(x), ϵ=ofeltype(x, 1e-5))
  μ = mean(x, dims=dims)
    #   σ = std(x, dims=dims, mean=μ, corrected=false) # use this when Zygote#478 gets merged
  σ = std(x, dims=dims, corrected=false)
  return (x .- μ) ./ (σ .+ ϵ)
end

"""
 cos_embedding_loss(x1, x2, y; margin=0, pad=false)

 Finds the cosine distance between `x1` and `x2` matrices for `y` = 1 or -1.
 `margin` can take values from [-1,1].
"""

function cos_embedding_loss(x1, x2, y; margin=0, pad=false) where N
    @assert (margin <= 1 && margin >= -1)
    @assert y ∈ (1,-1)
    if (pad == false && (size(x1) != size(x2)))
        throw(DimensionMismatch("If you wish to calculate the loss by padding zeros, pass 'pad = true and add PaddedViews.jl to the environment'"))
    elseif (pad == true && (size(x1) != size(x2)))
        size(x1) > size(x2) ? x2 = PaddedView(0.0, x2, size(x1)) : x1 = PaddedView(0.0, x1, (size(x2)))
    end
    if (y == 1)
        return (1 - (dot(x1,x2) / (norm(x1) * norm(x2))))
    elseif (y == -1)
        return max(0, (( dot(x1,x2) / (norm(x1) * norm(x2)))-margin))
    end
end
