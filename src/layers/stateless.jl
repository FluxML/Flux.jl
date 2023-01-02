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

julia> m = Chain(Conv((3,3), 3 => 4, pad=1), Flux.flatten, Dense(400 => 33));

julia> xs |> m[1] |> size
(10, 10, 4, 7)

julia> xs |> m |> size
(33, 7)
```
"""
function flatten(x::AbstractArray)
  return reshape(x, :, size(x)[end])
end

# Utils for LayerNorm internals.
# Most of these are required for better performance and type stability under AD.
# In an ideal world, we'd just have normalise.

function _mean_std(x::AbstractArray, dims)
  μ = mean(x, dims=dims)
  σ = std(x, dims=dims, mean=μ, corrected=false)
  return μ, σ
end

function ChainRulesCore.rrule(::typeof(_mean_std), x::AbstractArray, dims)
  μ, mean_pullback = ChainRulesCore.rrule(mean, x, dims=dims)
  σ, std_pullback = ChainRulesCore.rrule(std, x, dims=dims, mean=μ, corrected=false)
  function _mean_std_pullback((dμ, dσ))
    dx = ChainRulesCore.add!!(std_pullback(dσ)[2], mean_pullback(dμ)[2])
    return (NoTangent(), dx, NoTangent())
  end

  return (μ, σ), _mean_std_pullback
end

# We don't define a rrule for the whole function because we want
# AD to figure out the broadcast for us.
function _normalize(x::AbstractArray, dims, ϵ)
  μ, σ = _mean_std(x, dims)
  return @. (x - μ) / (σ + ϵ)
end

"""
    normalise(x; dims=ndims(x), ϵ=1e-5)

Normalise `x` to mean 0 and standard deviation 1 across the dimension(s) given by `dims`.
Per default, `dims` is the last dimension.
`ϵ` is a small additive factor added to the denominator for numerical stability.

# Examples
```jldoctest
julia> using Statistics

julia> x = [9, 10, 20, 60];

julia> y = Flux.normalise(x);

julia> isapprox(std(y), 1, atol=0.2) && std(y) != std(x)
true

julia> x = rand(1:100, 10, 2);

julia> y = Flux.normalise(x, dims=1);

julia> isapprox(std(y, dims=1), ones(1, 2), atol=0.2) && std(y, dims=1) != std(x, dims=1)
true
```
"""
@inline normalise(x::AbstractArray; dims=ndims(x), ϵ=ofeltype(x, 1e-5)) = _normalize(x, dims, ϵ)
