using Flux, Test, Random
using Random

Random.seed!(0)

@testset "Flux" begin

include("utils.jl")
include("tracker.jl")
include("layers/normalisation.jl")
include("layers/stateless.jl")
include("optimise.jl")
include("data.jl")

if Base.find_package("CuArrays") ≠ nothing
  include("cuda/cuda.jl")
end

end
