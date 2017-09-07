module Optimise

export update!, params, train!,
  SGD

include("params.jl")
include("optimisers.jl")
include("interface.jl")
include("train.jl")

using Flux.Tracker: TrackedArray

params(ps, p::TrackedArray) = push!(ps, p)

Base.convert(::Type{Param}, x::TrackedArray) = Param(x.data, x.grad[])

end
