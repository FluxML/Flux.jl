# v0.12 deprecations
@deprecate Dropout(p, dims) Dropout(p; dims=dims)
@deprecate InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum, active=nothing) InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum, true, true, active, length(β))
@deprecate BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, active=nothing) BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, true, true, active, length(β))
@deprecate GroupNorm(G, λ, β, γ, μ, σ², ϵ, momentum, active=nothing) GroupNorm(G, λ, β, γ, μ, σ², ϵ, momentum, true, true, active, length(β))
@deprecate outdims(f, inputsize) outputsize(f, inputsize)
@deprecate Conv(; weight,  bias, activation=identity, kws...) Conv(weight, bias, activation; kws...) 
@deprecate ConvTranspose(; weight, bias, activation=identity, kws...) ConvTranspose(weight, bias, activation; kws...) 
@deprecate DepthwiseConv(; weight, bias, activation=identity, kws...) DepthwiseConv(weight, bias, activation; kws...) 

function Base.getproperty(a::Dense, s::Symbol)
  if s === :W
    Base.depwarn("field name dense.W is deprecated in favour of dense.weight", :Dense)
    return getfield(a, :weight)
  elseif s === :b
    Base.depwarn("field name dense.b is deprecated in favour of dense.bias", :Dense)
    return getfield(a, :bias)
  end
  return getfield(a, s)
end

function ones(dims...)
  Base.depwarn("Flux.ones(size...) is deprecated, please use Flux.ones32(size...) or Base.ones(Float32, size...)", :ones)
end
ones(T::Type, dims...) = Base.ones(T, dims...) # no need for a message

function zeros(dims...)
  Base.depwarn("Flux.zeros(size...) is deprecated, please use Flux.zeros32(size...) or Base.zeros(Float32, size...)", :ones)
end
zeros(T::Type, dims...) = Base.zeros(T, dims...)

ones32(::Type, dims...) = throw(ArgumentError("Flux.ones32 is always Float32, use Base.ones to specify the element type"))
zeros32(::Type, dims...) = throw(ArgumentError("Flux.zeros32 is always Float32, use Base.zeros to specify the element type"))
