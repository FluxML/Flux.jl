# Gradient parity against Zygote for layers and layer options not in `TEST_MODELS`. The other layers
# are in conv_reactant.jl, pooling_reactant.jl, upsample_reactant.jl, normalisation_reactant.jl and
# sequence_reactant.jl, split up to keep each test file short.

@testset "Reactant basic layer gradients" begin
    r(dims...) = randn(Float32, dims...)
    # (model, inputs, name, keyword arguments for `test_gradients`)
    test_reactant_gradients([
        [(Dense(3 => 4, act), (r(3, 5),), "Dense($act)", (;))
         for act in (gelu, swish, elu, selu, mish, hardswish, softplus)]...,

        (Parallel(+, Dense(3 => 4), Dense(3 => 4, tanh)), (r(3, 5),), "Parallel(+)", (;)),
        (Parallel(vcat, Dense(3 => 4), Dense(2 => 4)), (r(3, 5), r(2, 5)), "Parallel(vcat), 2 inputs", (;)),
        (PairwiseFusion(+, Dense(3 => 3), Dense(3 => 3)), (r(3, 4), r(3, 4)), "PairwiseFusion",
         (; loss = (m, xs...) -> mean(sum(m(xs...))))),
        (Embedding(10 => 4), (rand(1:10, 3, 5),), "Embedding, Int input", (;)),
        (Embedding(10 => 4), (onehotbatch(rand(1:10, 5), 1:10),), "Embedding, onehot input", (;)),
        (Chain(Dense(3 => 4), softmax), (r(3, 5),), "Chain(Dense, softmax)", (;)),
        (Chain(Dense(3 => 4), logsoftmax), (r(3, 5),), "Chain(Dense, logsoftmax)", (;)),
        (Chain(Dense(3 => 4), Flux.flatten), (r(3, 2, 5),), "Chain(Dense, flatten)", (;)),

        # Dropout in train mode doesn't compile yet: https://github.com/FluxML/NNlib.jl/issues/807
        (Chain(Dense(3 => 4), Dropout(0.5)), (r(3, 5),), "Dropout, test mode", (; test_mode = true)),
        (Chain(Dense(3 => 4), AlphaDropout(0.5)), (r(3, 5),), "AlphaDropout, test mode", (; test_mode = true)),
    ])
end
