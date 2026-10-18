# `trainstep!` on a Reactant device follows the eager trajectory, for Adam and its variants.

@testset "Reactant Adam trajectories match eager" begin
    on_cuda = reactant_on_cuda()
    rtol = reactant_trajectory_rtol()

    rules = [
        Adam(0.01f0), AdaMax(0.01f0), OAdam(0.01f0), NAdam(0.01f0), AdamW(0.01f0, (0.9f0, 0.999f0), 0.1f0),
        AdaBelief(0.01f0),
        OptimiserChain(ClipNorm(0.5f0), Adam(0.01f0)),
        OptimiserChain(WeightDecay(0.1f0), Adam(0.01f0)),
    ]
    for rule in rules
        @testset "$rule" begin
            @test reactant_trajectory_error(rule) < rtol
        end
    end

    # `Flux.setup` fails on Reactant's GPU backend: https://github.com/FluxML/Optimisers.jl/issues/235
    @testset "AMSGrad" begin
        @test reactant_trajectory_error(AMSGrad(0.01f0)) < rtol  broken=on_cuda
    end
    # The step counter never advances: https://github.com/FluxML/Optimisers.jl/issues/234
    @testset "RAdam" begin
        @test_broken reactant_trajectory_error(RAdam(0.01f0)) < rtol
    end
    # Adam's state advances on every step, not once per accumulation: https://github.com/FluxML/Optimisers.jl/issues/237
    @testset "OptimiserChain(AccumGrad(3), Adam)" begin
        @test_broken reactant_trajectory_error(OptimiserChain(AccumGrad(3), Adam(0.01f0))) < rtol
    end
end
