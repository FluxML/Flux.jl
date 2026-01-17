@testset "mooncake gradient" begin
    for (model, x, name) in TEST_MODELS
        @testset "grad check $name" begin
            @test test_gradients(model, x; reference=AutoZygote(), compare=AutoMooncake())
        end
    end 
end
