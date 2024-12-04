@testset "params" begin
    ps = Flux.params([2,3])
    @test length(ps) == 1
end
