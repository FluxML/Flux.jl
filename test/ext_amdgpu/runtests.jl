
@assert AMDGPU.functional()
AMDGPU.allowscalar(false)

include("../test_utils.jl")
include("test_utils.jl")

@testset "get_devices" begin
  include("get_devices.jl")
end

@testset "Basic" begin
    include("basic.jl")
end
