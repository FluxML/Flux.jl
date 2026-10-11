if FLUX_TEST_REACTANT_CUDA
    @testset "Reactant runs on CUDA" begin
        @test Reactant.XLA.platform_name(Reactant.XLA.default_backend()) == "cuda"
    end
end

@testset "Reactant Models" begin
    # On top of the shared models: a loss reducing a conv output with more than one element
    # (the shared `Conv` case has a single output element), and the non-zero padding modes.
    reactant_models = [
        (Conv((3, 3), 2 => 3), randn(Float32, 6, 6, 2, 2), "Conv, multi-element output"),
        [(Conv((3, 3), 2 => 3; pad=1, pad_mode), randn(Float32, 6, 6, 2, 2), "Conv pad_mode=$(repr(pad_mode))")
         for pad_mode in (:circular, :reflect, :replicate, :symmetric)]...,
    ]
    broken_models = ()
    for (model, x, name) in vcat(TEST_MODELS, reactant_models)
        @testset "Reactant grad check $name" begin
            @test test_gradients(model, x; reference=AutoZygote(), test_reactant=true, test_cpu=false)  broken=(name ∈ broken_models)
        end
    end
end
