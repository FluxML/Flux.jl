@testset "Normalization" begin
    include("../test_common/normalization.jl")
    normalization_testsuite(Flux.gpu_device(force=true))
end
