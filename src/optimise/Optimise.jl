module Optimise

export update!, params, train!,
  SGD, ADAM, Momentum, Nesterov, RMSProp, ADAGrad, ADADelta

struct Param{T}
  x::T
  Δ::T
end

Base.convert(::Type{Param}, x::AbstractArray) = Param(x, zeros(x))

include("optimisers.jl")
include("interface.jl")
include("train.jl")

using Flux.Tracker: TrackedArray

Base.convert(::Type{Param}, x::TrackedArray) = Param(x.data, x.grad[])

end
