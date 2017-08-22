module Optimise

using ..Tracker: TrackedArray, data, grad

export sgd, update!, params

include("params.jl")
include("optimisers.jl")

end
