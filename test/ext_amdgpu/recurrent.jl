
@testset "Recurrent" begin
  global BROKEN_TESTS = []
  include("../ext_gpu_common/recurrent.jl")
end
