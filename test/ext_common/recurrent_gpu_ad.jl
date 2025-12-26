cell_loss(cell, x, state) = mean(cell(x, state)[1])

function recurrent_cell_loss(cell, seq, state)
    out = []
    for xt in seq
        yt, state = cell(xt, state)
        out = vcat(out, [yt])
    end
    return mean(stack(out, dims = 2))
end

@testset "RNNCell GPU AD" begin
    d_in, d_out, len, batch_size = 2, 3, 4, 5
    r = RNNCell(d_in => d_out)
    x = [randn(Float32, d_in, batch_size) for _ in 1:len]
    h = zeros(Float32, d_out)
    # Single Step
    @test test_gradients(r, x[1], h; test_gpu=true, 
            reference=AutoZygote(), compare=nothing, 
            loss=cell_loss) broken = :rnncell_single ∈ BROKEN_TESTS
    # Multiple Steps
    @test test_gradients(r, x, h; test_gpu=true, 
            reference=AutoZygote(), compare=nothing, 
            loss=recurrent_cell_loss)  broken = :rnncell_multiple ∈ BROKEN_TESTS
end

@testset "RNN GPU AD" begin
    struct ModelRNN
        rnn::RNN
        h0::AbstractVector
    end

    Flux.@layer ModelRNN

    (m::ModelRNN)(x) = m.rnn(x, m.h0)

    d_in, d_out, len, batch_size = 2, 3, 4, 5
    model = ModelRNN(RNN(d_in => d_out), zeros(Float32, d_out))
    x_nobatch = randn(Float32, d_in, len)
    @test test_gradients(model, x_nobatch; test_gpu=true,
        reference=AutoZygote(), compare=nothing)  broken = :rnn_nobatch ∈ BROKEN_TESTS
    x = randn(Float32, d_in, batch_size)
    @test test_gradients(model, x, test_gpu=true, 
        reference=AutoZygote(), compare=nothing)  broken = :rnn ∈ BROKEN_TESTS
end

@testset "LSTMCell" begin
    d_in, d_out, len, batch_size = 2, 3, 4, 5
    cell = LSTMCell(d_in => d_out)
    x = [randn(Float32, d_in, batch_size) for _ in 1:len]
    h = zeros(Float32, d_out)
    c = zeros(Float32, d_out)
    # Single Step
    @test test_gradients(cell, x[1], (h, c); test_gpu=true, 
    reference=AutoZygote(), compare=nothing,
        loss = cell_loss)  broken = :lstmcell_single ∈ BROKEN_TESTS
    # Multiple Steps
    @test test_gradients(cell, x, (h, c); test_gpu=true, 
        reference=AutoZygote(), compare=nothing,
        loss = recurrent_cell_loss)  broken = :lstmcell_multiple ∈ BROKEN_TESTS
end

@testset "LSTM" begin
    struct ModelLSTM
        lstm::LSTM
        h0::AbstractVector
        c0::AbstractVector
    end

    Flux.@layer ModelLSTM

    (m::ModelLSTM)(x) = m.lstm(x, (m.h0, m.c0))

    d_in, d_out, len, batch_size = 2, 3, 4, 5
    model = ModelLSTM(LSTM(d_in => d_out), zeros(Float32, d_out), zeros(Float32, d_out))
    x_nobatch = randn(Float32, d_in, len)
    @test test_gradients(model, x_nobatch; test_gpu=true, 
        reference=AutoZygote(), compare=nothing) broken = :lstm_nobatch ∈ BROKEN_TESTS
    x = randn(Float32, d_in, len, batch_size)
    @test test_gradients(model, x; test_gpu=true, 
        reference=AutoZygote(), compare=nothing) broken = :lstm ∈ BROKEN_TESTS
end

@testset "GRUCell" begin
    d_in, d_out, len, batch_size = 2, 3, 4, 5
    r = GRUCell(d_in => d_out)
    x = [randn(Float32, d_in, batch_size) for _ in 1:len]
    h = zeros(Float32, d_out)
    @test test_gradients(r, x[1], h; test_gpu=true, 
        reference=AutoZygote(), compare=nothing, 
        loss = cell_loss) broken = :grucell_single ∈ BROKEN_TESTS
    @test test_gradients(r, x, h; test_gpu=true, 
        reference=AutoZygote(), compare=nothing, 
        loss = recurrent_cell_loss) broken = :grucell_multiple ∈ BROKEN_TESTS
end

@testset "GRU GPU AD" begin
    struct ModelGRU
        gru::GRU
        h0::AbstractVector
    end

    Flux.@layer ModelGRU

    (m::ModelGRU)(x) = m.gru(x, m.h0)

    d_in, d_out, len, batch_size = 2, 3, 4, 5
    model = ModelGRU(GRU(d_in => d_out), zeros(Float32, d_out))
    x_nobatch = randn(Float32, d_in, len)
    @test test_gradients(model, x_nobatch; test_gpu=true, 
        reference=AutoZygote(), compare=nothing) broken = :gru_nobatch ∈ BROKEN_TESTS
    x = randn(Float32, d_in, len, batch_size)
    @test test_gradients(model, x; test_gpu=true, 
        reference=AutoZygote(), compare=nothing) broken = :gru ∈ BROKEN_TESTS
end

@testset "GRUv3Cell GPU AD" begin
    d_in, d_out, len, batch_size = 2, 3, 4, 5
    r = GRUv3Cell(d_in => d_out)
    x = [randn(Float32, d_in, batch_size) for _ in 1:len]
    h = zeros(Float32, d_out)
    @test test_gradients(r, x[1], h; test_gpu=true, 
        reference=AutoZygote(), compare=nothing,
        loss=cell_loss) broken = :gruv3cell_single ∈ BROKEN_TESTS
    @test test_gradients(r, x, h; test_gpu=true, 
        reference=AutoZygote(), compare=nothing, 
        loss = recurrent_cell_loss) broken = :gruv3cell_multiple ∈ BROKEN_TESTS
end

@testset "GRUv3 GPU AD" begin
    struct ModelGRUv3
        gru::GRUv3
        h0::AbstractVector
    end

    Flux.@layer ModelGRUv3

    (m::ModelGRUv3)(x) = m.gru(x, m.h0)
    
    d_in, d_out, len, batch_size = 2, 3, 4, 5
    model = ModelGRUv3(GRUv3(d_in => d_out), zeros(Float32, d_out))
    x_nobatch = randn(Float32, d_in, len)
    @test test_gradients(model, x_nobatch; test_gpu=true, 
        reference=AutoZygote(), compare=nothing) broken = :gruv3_nobatch ∈ BROKEN_TESTS
    x = randn(Float32, d_in, len, batch_size)
    @test test_gradients(model, x; test_gpu=true, 
        reference=AutoZygote(), compare=nothing) broken = :gruv3 ∈ BROKEN_TESTS
end
