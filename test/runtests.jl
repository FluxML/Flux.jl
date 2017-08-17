using Flux, DataFlow, MacroTools, Base.Test
using Flux: graph, Param, squeeze, unsqueeze, back!, update!, flatten
using DataFlow: Line, Frame

@testset "Flux" begin

include("batching.jl")
include("backend/common.jl")

include("basic.jl")
include("recurrent.jl")
include("optimizer.jl")
include("throttle.jl")

end
