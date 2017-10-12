using Flux.Optimise
using Flux.Tracker

@testset "Optimise" begin
    loss(x) = sum(x.^2)
    η = 0.1
    # RMSProp gets stuck
    for OPT in [SGD, Nesterov, Momentum, ADAM, ADAGrad, ADADelta]
        x = param(randn(10))
        opt = OPT == ADADelta ? OPT([x]) : OPT([x], η)
        for t=1:10000
            l = loss(x)
            back!(l)
            opt()
            l.data[] < 1e-10 && break
        end
        @test loss(x) ≈ 0. atol=1e-7
    end
end
