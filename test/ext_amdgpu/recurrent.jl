
@testset "Recurrent" begin
  global BROKEN_TESTS = []
  include("../test_common/gpu_recurrent.jl")
end
