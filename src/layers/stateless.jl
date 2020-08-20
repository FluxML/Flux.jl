"""
    flatten(x::AbstractArray)

Reshape arbitrarly-shaped input into a matrix-shaped output
preserving the last dimension size. 
Equivalent to `reshape(x, :, size(x)[end])`.
"""
function flatten(x::AbstractArray)
  return reshape(x, :, size(x)[end])
end

# TODO normalise over last dimension is typically what you want to do. 
# Deprecation path: `normalise(x; dims=1)` -> `normalise(x; dims)` -> `normalise(x; dims=size(x)[end])`  
"""
    normalise(x; dims, ϵ=1e-5)

Normalise `x` to mean 0 and standard deviation 1 across the dimensions given by `dims`.
`ϵ` is a small additive factor added to the denominator for numerical stability.
"""
function normalise(x::AbstractArray; dims, ϵ=ofeltype(x, 1e-5))
  μ = mean(x, dims=dims)
    #   σ = std(x, dims=dims, mean=μ, corrected=false) # use this when #478 gets merged
  σ = std(x, dims=dims, corrected=false)
  return (x .- μ) ./ (σ .+ ϵ)
end