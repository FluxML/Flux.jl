@testset "Reactant recurrent and attention layer gradients" begin
    r(dims...) = randn(Float32, dims...)
    first_mean(m, xs...) = mean(first(m(xs...)))
    # (model, inputs, name, keyword arguments for `test_gradients`), sequences as (features, seq, batch)
    test_reactant_gradients([
        (RNN(3 => 4), (r(3, 6, 2),), "RNN sequence", (;)),
        (LSTM(3 => 4), (r(3, 6, 2),), "LSTM sequence", (;)),
        (GRU(3 => 4), (r(3, 6, 2),), "GRU sequence", (;)),
        (Chain(Dense(3 => 4), LSTM(4 => 5)), (r(3, 6, 2),), "Chain(Dense, LSTM)", (;)),
        (LSTMCell(3 => 4), (r(3, 5),), "LSTMCell", (; loss = first_mean)),
        (GRUCell(3 => 4), (r(3, 5),), "GRUCell", (; loss = first_mean)),
        (Attention(16; nheads = 2), (r(16, 7, 2),), "Attention(16; nheads=2)", (;)),
    ])
end
