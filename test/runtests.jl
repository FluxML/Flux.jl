using Flux, Base.Test

@testset "Flux" begin

include("utils.jl")
include("tracker.jl")
include("layers/normalisation.jl")
include("layers/stateless.jl")
include("optimise.jl")
include("data.jl")

if Base.find_in_path("CuArrays") ≠ nothing
  include("cuarrays.jl")
end

end
