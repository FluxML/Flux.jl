include("utils.jl")

AMDGPU.allowscalar(false)

@testset "Basic" begin
    include("basic.jl")
end
