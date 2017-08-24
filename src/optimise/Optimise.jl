module Optimise

using ..Tracker: TrackedArray, data, grad, back!

export sgd, update!, params, train!

include("params.jl")
include("optimisers.jl")
include("train.jl")

end
