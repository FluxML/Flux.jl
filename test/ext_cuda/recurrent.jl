
@testset "Recurrent" begin
  global BROKEN_TESTS = []
  include("../ext_common/recurrent_gpu_ad.jl")
end

