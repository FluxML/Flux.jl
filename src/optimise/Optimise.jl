module Optimise

export train!,
  SGD, ADAM, ADAMW, AdaMax, Momentum, Nesterov,
  RMSProp, ADAGrad, ADADelta, AMSGrad, NADAM

struct Param{T}
  x::T
  Δ::T
end

Param(x::AbstractArray) = Param(x, zero(x))

include("optimisers.jl")
include("interface.jl")
include("train.jl")

using Flux.Tracker: TrackedArray

Param(x::TrackedArray) = Param(x.data, x.grad)
# Base.convert(::Type{Param}, x::TrackedArray) = Param(x.data, x.grad)

end
