# `trainstep!` on a Reactant device follows the eager trajectory. Adam and its variants are in
# optimisers_adam_reactant.jl.

@testset "Reactant optimiser trajectories match eager" begin
    on_cuda = reactant_on_cuda()
    rtol = reactant_trajectory_rtol()

    rules = [
        Descent(0.1f0), Momentum(0.05f0), Nesterov(0.05f0), RMSProp(0.01f0), AdaDelta(), Lion(0.01f0),
        OptimiserChain(ClipGrad(0.05f0), Descent(0.1f0)),
    ]
    for rule in rules
        @testset "$rule" begin
            @test reactant_trajectory_error(rule) < rtol
        end
    end

    # `Flux.setup` fails on Reactant's GPU backend: https://github.com/FluxML/Optimisers.jl/issues/235
    @testset "AdaGrad" begin
        @test reactant_trajectory_error(AdaGrad(0.05f0)) < rtol  broken=on_cuda
    end
    # Fails to compile: https://github.com/FluxML/Optimisers.jl/issues/236
    @testset "Rprop" begin
        @test_broken reactant_trajectory_error(Rprop(0.01f0)) < rtol
    end
end
