
@testset "RNNCell GPU AD" begin
    function loss(r, h, x)
        y = []
        for x_t in x
            h = r(h, x_t)
            y = vcat(y, [h])
        end
        # return mean(h)
        y = stack(y, dims=2) # [D, L] or [D, L, B]
        return mean(y)
    end

    d_in, d_out, len, batch_size = 2, 3, 4, 5
    r = RNNCell(d_in => d_out)
    x = [randn(Float32, d_in, batch_size) for _ in 1:len]
    h = zeros(Float32, d_out)
    # Single Step
    @test test_gradients(r, x[1], h; test_gpu=true, compare_finite_diff=false) broken = :rnncell_single ∈ BROKEN_TESTS
    # Multiple Steps
    @test test_gradients(r, h, x; test_gpu=true, compare_finite_diff=false, loss)  broken = :rnncell_multiple ∈ BROKEN_TESTS
end

@testset "RNN GPU AD" begin
    struct ModelRNN
        rnn::RNN
        h0::AbstractVector
    end

    Flux.@layer :expand ModelRNN

    (m::ModelRNN)(x) = m.rnn(m.h0, x)

    d_in, d_out, len, batch_size = 2, 3, 4, 5
    model = ModelRNN(RNN(d_in => d_out), zeros(Float32, d_out))
    x_nobatch = randn(Float32, d_in, len)
    @test test_gradients(model, x_nobatch; test_gpu=true, compare_finite_diff=false)  broken = :rnn_nobatch ∈ BROKEN_TESTS
    x = randn(Float32, d_in, batch_size)
    @test test_gradients(model, x, test_gpu=true, compare_finite_diff=false)  broken = :rnn ∈ BROKEN_TESTS
end

@testset "LSTMCell" begin

    function loss(r, hc, x)
        h, c = hc
        h′ = []
        c′ = []
        for x_t in x
            h, c = r((h, c), x_t)
            h′ = vcat(h′, [h])
            c′ = [c′..., c]
        end
        hnew = stack(h′, dims=2)
        cnew = stack(c′, dims=2)
        return mean(hnew) + mean(cnew)
    end

    d_in, d_out, len, batch_size = 2, 3, 4, 5
    cell = LSTMCell(d_in => d_out)
    x = [randn(Float32, d_in, batch_size) for _ in 1:len]
    h = zeros(Float32, d_out)
    c = zeros(Float32, d_out)
    # Single Step
    @test test_gradients(cell, x[1], (h, c); test_gpu=true, compare_finite_diff=false,
        loss = (m, (h, c), x) -> mean(m((h, c), x)[1]))  broken = :lstmcell_single ∈ BROKEN_TESTS
    # Multiple Steps
    @test test_gradients(cell, (h, c), x; test_gpu=true, compare_finite_diff=false, loss)  broken = :lstmcell_multiple ∈ BROKEN_TESTS
end

@testset "LSTM" begin
    struct ModelLSTM
        lstm::LSTM
        h0::AbstractVector
        c0::AbstractVector
    end

    Flux.@layer :expand ModelLSTM

    (m::ModelLSTM)(x) = m.lstm(x, (m.h0, m.c0))

    d_in, d_out, len, batch_size = 2, 3, 4, 5
    model = ModelLSTM(LSTM(d_in => d_out), zeros(Float32, d_out), zeros(Float32, d_out))
    x_nobatch = randn(Float32, d_in, len)
    @test test_gradients(model, x_nobatch; test_gpu=true, compare_finite_diff=false, 
        loss = (m, x) -> mean(m(x)[1])) broken = :lstm_nobatch ∈ BROKEN_TESTS
    x = randn(Float32, d_in, len, batch_size)
    @test test_gradients(model, x; test_gpu=true, compare_finite_diff=false, 
        loss = (m, x) -> mean(m(x)[1])) broken = :lstm ∈ BROKEN_TESTS
end

@testset "GRUCell" begin
    function loss(r, h, x)
        y = []
        for x_t in x
            h = r(h, x_t)
            y = vcat(y, [h])
        end
        y = stack(y, dims=2) # [D, L] or [D, L, B]
        return mean(y)
    end

    d_in, d_out, len, batch_size = 2, 3, 4, 5
    r = GRUCell(d_in => d_out)
    x = [randn(Float32, d_in, batch_size) for _ in 1:len]
    h = zeros(Float32, d_out)
    @test test_gradients(r, x[1], h; test_gpu=true, compare_finite_diff=false) broken = :grucell_single ∈ BROKEN_TESTS
    @test test_gradients(r, h, x; test_gpu=true, compare_finite_diff=false, loss) broken = :grucell_multiple ∈ BROKEN_TESTS
end

@testset "GRU GPU AD" begin
    struct ModelGRU
        gru::GRU
        h0::AbstractVector
    end

    Flux.@layer :expand ModelGRU

    (m::ModelGRU)(x) = m.gru(m.h0, x)

    d_in, d_out, len, batch_size = 2, 3, 4, 5
    model = ModelGRU(GRU(d_in => d_out), zeros(Float32, d_out))
    x_nobatch = randn(Float32, d_in, len)
    @test test_gradients(model, x_nobatch; test_gpu=true, compare_finite_diff=false) broken = :gru_nobatch ∈ BROKEN_TESTS
    x = randn(Float32, d_in, len, batch_size)
    @test test_gradients(model, x; test_gpu=true, compare_finite_diff=false) broken = :gru ∈ BROKEN_TESTS
end

@testset "GRUv3Cell GPU AD" begin
    function loss(r, h, x)
        y = []
        for x_t in x
            h = r(h, x_t)
            y = vcat(y, [h])
        end
        y = stack(y, dims=2) # [D, L] or [D, L, B]
        return mean(y)
    end

    d_in, d_out, len, batch_size = 2, 3, 4, 5
    r = GRUv3Cell(d_in => d_out)
    x = [randn(Float32, d_in, batch_size) for _ in 1:len]
    h = zeros(Float32, d_out)
    @test test_gradients(r, x[1], h; test_gpu=true, compare_finite_diff=false) broken = :gruv3cell_single ∈ BROKEN_TESTS
    @test test_gradients(r, h, x; test_gpu=true, compare_finite_diff=false, loss) broken = :gruv3cell_multiple ∈ BROKEN_TESTS
end

@testset "GRUv3 GPU AD" begin
    struct ModelGRUv3
        gru::GRUv3
        h0::AbstractVector
    end

    Flux.@layer :expand ModelGRUv3

    (m::ModelGRUv3)(x) = m.gru(m.h0, x)
    
    d_in, d_out, len, batch_size = 2, 3, 4, 5
    model = ModelGRUv3(GRUv3(d_in => d_out), zeros(Float32, d_out))
    x_nobatch = randn(Float32, d_in, len)
    @test test_gradients(model, x_nobatch; test_gpu=true, compare_finite_diff=false) broken = :gruv3_nobatch ∈ BROKEN_TESTS
    x = randn(Float32, d_in, len, batch_size)
    @test test_gradients(model, x; test_gpu=true, compare_finite_diff=false) broken = :gruv3 ∈ BROKEN_TESTS
end
