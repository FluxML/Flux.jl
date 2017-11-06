module Data

import ..Flux

export CMUDict, cmudict

deps(path...) = joinpath(@__DIR__, "..", "..", "deps", path...)

function __init__()
  mkpath(deps())
end

include("cmudict.jl")
using .CMUDict

include("sentiment.jl")
using .Sentiment

end
