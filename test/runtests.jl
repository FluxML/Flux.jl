using Flux, Test, Random
using Random

Random.seed!(0)

# So we can use the system CuArrays
insert!(LOAD_PATH, 2, "@v#.#")

@testset "Flux" begin

include("utils.jl")
include("tracker.jl")
include("layers/normalisation.jl")
include("layers/stateless.jl")
include("optimise.jl")
include("data.jl")

if Base.find_package("CuArrays") != nothing
  include("cuda/cuda.jl")
end

end
