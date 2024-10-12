
@assert AMDGPU.functional()
AMDGPU.allowscalar(false)

@testset "get_devices" begin
  include("get_devices.jl")
end

@testset "Basic" begin
    include("basic.jl")
end
