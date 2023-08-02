using Test
using Metal
using Flux
using Random, Statistics
using Zygote
Flux.gpu_backend!("Metal") # needs a restart

# include("../test_utils.jl")
include("test_utils.jl")

@testset "Basic" begin
    include("basic.jl")
end

@testset "get_device" begin
    include("get_device.jl")
end
