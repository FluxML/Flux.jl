@testset "Reactant Models" begin
    for (model, x, name) in TEST_MODELS
        @testset "Reactant grad check $name" begin
            @test test_gradients(model, x; reference=AutoZygote(), test_reactant=true, test_cpu=false)
        end
    end
end
