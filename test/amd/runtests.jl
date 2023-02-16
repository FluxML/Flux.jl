Flux.gpu_backend!("AMD")

include("utils.jl")

AMDGPU.allowscalar(false)

@testset "Basic" begin
    include("basic.jl")
end
