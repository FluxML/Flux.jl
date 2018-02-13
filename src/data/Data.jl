module Data

import ..Flux

export CMUDict, cmudict

deps(path...) = joinpath(@__DIR__, "..", "..", "deps", path...)

function __init__()
  mkpath(deps())
end

include("mnist.jl")
include("cmudict.jl")
using .CMUDict

include("tree.jl")
include("sentiment.jl")
using .Sentiment

end
