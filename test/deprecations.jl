@testset "params" begin
    ps = Flux.params([2,3])
    @test length(ps) == 1
end

@testset "Flux.@functor" begin
    # https://github.com/FluxML/Flux.jl/issues/2545
    struct A2545; x; y; end
    Flux.@functor A2545
    a = A2545(1, 2)
    @test fmap(x -> 2x, a) == A2545(2, 4)

    struct B2545; x; y; end
    Flux.@functor B2545 (x,)
    b = B2545(1, 2)
    @test fmap(x -> 2x, b) == B2545(2, 2)
end