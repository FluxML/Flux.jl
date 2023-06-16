
@assert AMDGPU.functional()
AMDGPU.allowscalar(false)

include("../test_utils.jl")
include("test_utils.jl")

@testset "Basic" begin
    include("basic.jl")
end
