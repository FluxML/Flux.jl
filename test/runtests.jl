using Flux, Test, Random
using Random

Random.seed!(0)

@testset "Flux" begin

println("Testing")
include("utils.jl")
# println("Testing")
# include("tracker.jl")
println("Testing")
include("layers/normalisation.jl")
println("Testing")
include("layers/stateless.jl")
println("Testing")
include("optimise.jl")
println("Testing")
include("data.jl")

# if Base.find_in_path("CuArrays") ≠ nothing
#   include("cuda/cuda.jl")
# end

end
