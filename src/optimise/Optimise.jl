module Optimise

export sgd, update!, params, train!

include("params.jl")
include("optimisers.jl")
include("train.jl")

using Flux.Tracker: TrackedArray

params(ps, p::TrackedArray) = push!(ps, p)

Base.convert(::Type{Param}, x::TrackedArray) = Param(x.x, x.Δ)

end
