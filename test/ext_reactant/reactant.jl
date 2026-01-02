function scalarfirst(x)
    Reactant.@allowscalar first(x)
end

@testset "Reactant Models" begin
    function loss(model, x)
        mean(model(x))
    end

    models_xs = [
        (Dense(2=>4), randn(Float32, 2), "Dense"),
        
        (Chain(Dense(2=>4, tanh), Dense(4=>3)), randn(Float32, 2), "Chain(Dense, Dense)"),

        (f64(Chain(Dense(2=>4), Dense(4=>2))), randn(Float64, 2, 1), "f64(Chain(Dense, Dense))"),

        (Flux.Scale([1.0f0 2.0f0 3.0f0 4.0f0], true, abs2), randn(Float32, 2), "Flux.Scale"),

        (Conv((3, 3), 2 => 3), randn(Float32, 3, 3, 2, 1), "Conv"),

        (Chain(Conv((3, 3), 2 => 3, ), Conv((3, 3), 3 => 1, tanh)), rand(Float32, 5, 5, 2, 1), "Chain(Conv, Conv)"),

        (Chain(Conv((4, 4), 2 => 2, pad=SamePad()), MeanPool((5, 5), pad=SamePad())), rand(Float32, 5, 5, 2, 2), "Chain(Conv, MeanPool)"), 
        
        (Maxout(() -> Dense(5 => 4, tanh), 3), randn(Float32, 5, 1), "Maxout"),

        (SkipConnection(Dense(2 => 2), vcat), randn(Float32, 2, 3), "SkipConnection"),
        
        (Flux.Bilinear((2, 2) => 3), randn(Float32, 2, 1), "Bilinear"),

        (ConvTranspose((3, 3), 3 => 2, stride=2), rand(Float32, 5, 5, 3, 1), "ConvTranspose"),

        (BatchNorm(2), randn(Float32, 2, 10), "BatchNorm"),
    ]

    for (model, x, name) in models_xs
        @testset "Enzyme grad check $name" begin
            println("testing $name with Reactant")
            test_gradients(model, x; loss, compare_finite_diff=false, test_reactant=true)
        end
    end

    models_xs = [
        (LayerNorm(2), randn(Float32, 2, 10), "LayerNorm"), # Zygote comparison test fails on the GPUArraysCore.@allowscalar in scalarfirst, so we globally allow scalar

        (first ∘ MultiHeadAttention(16), randn32(16, 20, 2), "MultiHeadAttention"), # Zygote comparison test fails on the GPUArraysCore.@allowscalar in scalarfirst, so we globally allow scalar
    ]

    Reactant.allowscalar(true)
    for (model, x, name) in models_xs
        @testset "Enzyme grad check $name" begin
            println("testing $name with Reactant")
            test_gradients(model, x; loss, compare_finite_diff=false, test_reactant=true)
        end
    end
    Reactant.allowscalar(false)
end

@testset "Reactant Recurrent Layers" begin
    function loss(model, x)
        mean(model(x))
    end

    models_xs = [
        (RNN(3 => 2), randn(Float32, 3, 2), "RNN"), 
        (LSTM(3 => 5), randn(Float32, 3, 2), "LSTM"),
        (GRU(3 => 5), randn(Float32, 3, 10), "GRU"),
        (Chain(RNN(3 => 4), RNN(4 => 3)), randn(Float32, 3, 2), "Chain(RNN, RNN)"),
        (Chain(LSTM(3 => 5), LSTM(5 => 3)), randn(Float32, 3, 2), "Chain(LSTM, LSTM)"),
    ]

    for (model, x, name) in models_xs
        @testset "check grad $name" begin
            println("testing $name with Reactant")
            test_gradients(model, x; loss, compare_finite_diff=false, test_reactant=true)
        end
    end
end
