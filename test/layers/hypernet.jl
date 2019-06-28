using Flux
using Flux: destructure,restructure
using Test


@testset "Correct Dimensions for HyperDense" begin
    m = Chain(Dense(4,5),Dense(5,2))
    ps, re = destructure(m)
    hyper = Chain(Dense(1,10,tanh),HyperDense(10,m))
    @test length(hyper([1])) == length(ps)
    hyper = Chain(Dense(1,10,tanh),HyperDense(10,m,tanh))
    @test length(hyper([1])) == length(ps)
end

@testset "Define HyperNet for Chain" begin
    m = Chain(Dense(5,10,σ),Dense(10,2))
    h = Chain(Dense(1,10,tanh),HyperDense(10,m))
    hyper = HyperNet(h,m)
    x = rand(5)
    @test restructure(m,h([1]))(x) == hyper([1])(x)
end
