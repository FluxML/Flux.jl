# Gradient parity against Zygote for layers and layer options not in `TEST_MODELS`; the convolutional
# ones are in conv_reactant.jl. Each case chains several layers, as every case costs two compilations.

@testset "Reactant layer gradients" begin
    r(dims...) = randn(Float32, dims...)
    acts = (gelu, swish, elu, selu, mish, hardswish, softplus)
    # (model, inputs, name, keyword arguments for `test_gradients`)
    test_reactant_gradients([
        (Chain([Dense(4 => 4, act) for act in acts]...), (r(4, 5),), "Dense with $(join(acts, ", "))", (;)),
        (Chain(Dense(3 => 4), Flux.flatten, Dense(8 => 4), softmax, Dense(4 => 3), logsoftmax), (r(3, 2, 5),),
         "Chain(Dense, flatten, softmax, logsoftmax)", (;)),
        (Parallel(vcat, Dense(3 => 4), Dense(2 => 4, tanh)), (r(3, 5), r(2, 5)), "Parallel(vcat), 2 inputs", (;)),
        (PairwiseFusion(+, Dense(3 => 3), Dense(3 => 3)), (r(3, 4), r(3, 4)), "PairwiseFusion",
         (; loss = (m, xs...) -> mean(sum(m(xs...))))),
        # A 2-D onehot input: reshaping a traced N-d one fails
        # (https://github.com/EnzymeAD/Reactant.jl/issues/3504).
        (Parallel(vcat, Chain(Embedding(10 => 4), Flux.flatten), Embedding(10 => 4)),
         (rand(1:10, 3, 5), onehotbatch(rand(1:10, 5), 1:10)), "Embedding, Int matrix and onehot inputs", (;)),
        # Only in test mode: the train-mode masks are random, see dropout_reactant.jl.
        (Chain(Dense(3 => 4), Dropout(0.5), AlphaDropout(0.5)), (r(3, 5),), "Dropout, AlphaDropout, test mode",
         (; test_mode = true)),

        [(Chain(Conv((3, 3), 2 => 4), BatchNorm(4), InstanceNorm(4), GroupNorm(4, 2)), (r(6, 6, 2, 4),),
          "BatchNorm, InstanceNorm, GroupNorm" * (test_mode ? ", test mode" : ""), (; test_mode))
         for test_mode in (false, true)]...,
        (Chain(LayerNorm((4, 3)), Flux.flatten, WeightNorm(Dense(12 => 4))), (r(4, 3, 5),),
         "LayerNorm((4, 3)), WeightNorm(Dense)", (;)),

        # sequences as (features, seq, batch)
        (Chain(Dense(3 => 4), RNN(4 => 4), LSTM(4 => 5), GRU(5 => 4)), (r(3, 6, 2),), "Chain(Dense, RNN, LSTM, GRU)", (;)),
        (Chain(LSTMCell(3 => 4), first, GRUCell(4 => 5), first), (r(3, 5),), "Chain(LSTMCell, GRUCell)", (;)),
        (Attention(16; nheads = 2), (r(16, 7, 2),), "Attention(16; nheads=2)", (;)),
    ])
end
