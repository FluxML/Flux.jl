@testset "Reactant Models" begin
    broken_models = ("BatchNorm",) # https://github.com/EnzymeAD/Reactant.jl/issues/2118
    for (model, x, name) in TEST_MODELS
        @testset "Reactant grad check $name" begin
            @test test_gradients(model, x; reference=AutoZygote(), test_reactant=true, test_cpu=false)  broken=(name ∈ broken_models)
        end
    end
end
