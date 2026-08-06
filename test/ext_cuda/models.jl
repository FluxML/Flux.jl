@testset "models' gradients on CUDA" begin
    for (model, x, name) in TEST_MODELS
        println("$name")
        @testset "Zygote grad check $name" begin
            @test test_gradients(model, x; test_gpu=true, test_cpu=false,
                    reference=AutoZygote(), compare=AutoZygote())
        end
        @testset "Mooncake grad check $name" begin
            @test test_gradients(model, x; test_gpu=true, test_cpu=false,
                    reference=AutoZygote(), compare=AutoMooncake())
        end
    end
end
