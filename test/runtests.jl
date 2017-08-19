using Flux, DataFlow, MacroTools, Base.Test
using Flux: Param, param, squeeze, unsqueeze, stack, update!, flatten
using Flux.Compiler: @net
using DataFlow: Line, Frame

@testset "Flux" begin

include("backend/common.jl")

include("basic.jl")
include("recurrent.jl")
include("throttle.jl")

end
