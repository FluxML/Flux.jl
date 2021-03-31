# v0.12 deprecations
@deprecate Dropout(p, dims) Dropout(p; dims=dims)
@deprecate InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum, active=nothing) InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum, true, true, active, length(β))
@deprecate BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, active=nothing) BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, true, true, active, length(β))
@deprecate GroupNorm(G, λ, β, γ, μ, σ², ϵ, momentum, active=nothing) GroupNorm(G, λ, β, γ, μ, σ², ϵ, momentum, true, true, active, length(β))
@deprecate outdims(f, inputsize) outputsize(f, inputsize)
@deprecate Conv(; weight,  bias, activation=identity, kws...) Conv(weight, bias, activation; kws...) 
@deprecate ConvTranspose(; weight, bias, activation=identity, kws...) ConvTranspose(weight, bias, activation; kws...) 
@deprecate DepthwiseConv(; weight, bias, activation=identity, kws...) DepthwiseConv(weight, bias, activation; kws...) 

# function Base.getproperty(a::Dense, s::Symbol)
#   if s === :W
#     Base.depwarn("field name dense.W is deprecated in favour of dense.weight", :Dense)
#     return getfield(a, :weight)
#   elseif s === :b
#     Base.depwarn("field name dense.b is deprecated in favour of dense.bias", :Dense)
#     return getfield(a, :bias)
#   end
#   return getfield(a, s)
# end
