
@assert AMDGPU.functional()
AMDGPU.allowscalar(false)

@testset "get_devices" begin
  include("get_devices.jl")
end

@testset "Basic" begin
    include("basic.jl")
end

@testset "Recurrent" begin
  global BROKEN_TESTS = []
  include("../ext_common/recurrent_gpu_ad.jl")
end
