using Flux, Test, Random, Statistics
using Random

Random.seed!(0)

# So we can use the system CuArrays
insert!(LOAD_PATH, 2, "@v#.#")

@testset "Flux" begin

@info "Testing Basics"

include("utils.jl")
include("onehot.jl")
include("optimise.jl")
include("data.jl")

@info "Testing Layers"

include("layers/basic.jl")
include("layers/normalisation.jl")
include("layers/stateless.jl")
include("layers/conv.jl")

@info "Running Gradient Checks"

include("tracker.jl")

if isdefined(Flux, :CUDA)
  include("cuda/cuda.jl")
else
  @warn "CUDA unavailable, not testing GPU support"
end

end
