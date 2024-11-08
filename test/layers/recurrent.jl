
@testset "RNNCell" begin
    function loss1(r, h, x)
        for x_t in x
            h = r(h, x_t)
        end
        return mean(h.^2)
    end

    function loss2(r, h, x)
        y = [r(h, x_t) for x_t in x]
        return sum(mean, y)
    end

    function loss3(r, h, x)
        y = []
        for x_t in x
            h = r(h, x_t)
            y = [y..., h]
        end
        return sum(mean, y)
    end

    function loss4(r, h, x)
        y = []
        for x_t in x
            h = r(h, x_t)
            y = vcat(y, [h])
        end
        y = stack(y, dims=2) # [D, L] or [D, L, B]
        return mean(y.^2)
    end

    r = RNNCell(3 => 5)
    @test length(Flux.trainables(r)) == 3
    # An input sequence of length 6 and batch size 4.
    x = [rand(Float32, 3, 4) for _ in 1:6]

    # Initial State is a single vector
    h = randn(Float32, 5)
    test_gradients(r, h, x, loss=loss1) # for loop
    test_gradients(r, h, x, loss=loss2) # comprehension
    test_gradients(r, h, x, loss=loss3) # splat
    test_gradients(r, h, x, loss=loss4) # vcat and stack

    # no initial state same as zero initial state
    @test r(x[1]) ≈ r(x[1], zeros(Float32, 5))

    # Now initial state has a batch dimension.
    h = randn(Float32, 5, 4)
    test_gradients(r, h, x, loss=loss4)

    # The input sequence has no batch dimension.
    x = [rand(Float32, 3) for _ in 1:6]
    h = randn(Float32, 5)
    test_gradients(r, h, x, loss=loss4)

    
    # No Bias 
    r = RNNCell(3 => 5, bias=false)
    @test length(Flux.trainables(r)) == 2
    test_gradients(r, h, x, loss=loss4)
end

@testset "RNN" begin
    struct ModelRNN
        rnn::RNN
        h0::AbstractVector
    end

    Flux.@layer :expand ModelRNN

    (m::ModelRNN)(x) = m.rnn(m.h0, x)

    model = ModelRNN(RNN(2 => 4), zeros(Float32, 4))
    
    x = rand(Float32, 2, 3, 1)
    y = model(x)
    @test y isa Array{Float32, 3}
    @test size(y) == (4, 3, 1)
    test_gradients(model, x)

    # no initial state same as zero initial state
    rnn = model.rnn 
    @test rnn(x) ≈ rnn(x, zeros(Float32, 4))

    x = rand(Float32, 2, 3)
    y = model(x)
    @test y isa Array{Float32, 2}
    @test size(y) == (4, 3)
    test_gradients(model, x)
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
        return mean(hnew.^2) + mean(cnew.^2)
    end

    cell = LSTMCell(3 => 5)
    @test length(Flux.trainables(cell)) == 3
    x = [rand(Float32, 3, 4) for _ in 1:6]
    h = zeros(Float32, 5, 4)
    c = zeros(Float32, 5, 4)
    hnew, cnew = cell(x[1], (h, c))
    @test hnew isa Matrix{Float32}
    @test cnew isa Matrix{Float32}
    @test size(hnew) == (5, 4)
    @test size(cnew) == (5, 4)
    test_gradients(cell, x[1], (h, c), loss = (m, hc, x) -> mean(m(hc, x)[1]))
    test_gradients(cell, (h, c), x, loss = loss)

    # no initial state same as zero initial state
    hnew1, cnew1 = cell(x[1])
    hnew2, cnew2 = cell(x[1], (zeros(Float32, 5), zeros(Float32, 5)))
    @test hnew1 ≈ hnew2
    @test cnew1 ≈ cnew2

    # no bias
    cell = LSTMCell(3 => 5, bias=false)
    @test length(Flux.trainables(cell)) == 2
end

@testset "LSTM" begin
    struct ModelLSTM
        lstm::LSTM
        h0::AbstractVector
        c0::AbstractVector
    end

    Flux.@layer :expand ModelLSTM

    (m::ModelLSTM)(x) = m.lstm(x, (m.h0, m.c0))

    model = ModelLSTM(LSTM(2 => 4), zeros(Float32, 4), zeros(Float32, 4))
    
    x = rand(Float32, 2, 3, 1)
    h, c = model(x)
    @test h isa Array{Float32, 3}
    @test size(h) == (4, 3, 1)
    @test c isa Array{Float32, 3}
    @test size(c) == (4, 3, 1)
    test_gradients(model, x, loss = (m, x) -> mean(m(x)[1]))

    x = rand(Float32, 2, 3)
    h, c = model(x)
    @test h isa Array{Float32, 2}
    @test size(h) == (4, 3)
    @test c isa Array{Float32, 2}
    @test size(c) == (4, 3)
    test_gradients(model, x, loss = (m, x) -> mean(m(x)[1]))
end

@testset "GRUCell" begin
    function loss(r, h, x)
        y = []
        for x_t in x
            h = r(h, x_t)
            y = vcat(y, [h])
        end
        y = stack(y, dims=2) # [D, L] or [D, L, B]
        return mean(y.^2)
    end

    r = GRUCell(3 => 5)
    @test length(Flux.trainables(r)) == 3
    # An input sequence of length 6 and batch size 4.
    x = [rand(Float32, 3, 4) for _ in 1:6]

    # Initial State is a single vector
    h = randn(Float32, 5)
    test_gradients(r, h, x; loss)

    # no initial state same as zero initial state
    @test r(x[1]) ≈ r(x[1], zeros(Float32, 5))

    # Now initial state has a batch dimension.
    h = randn(Float32, 5, 4)
    test_gradients(r, h, x; loss)

    # The input sequence has no batch dimension.
    x = [rand(Float32, 3) for _ in 1:6]
    h = randn(Float32, 5)
    test_gradients(r, h, x; loss)

    # No Bias 
    r = GRUCell(3 => 5, bias=false)
    @test length(Flux.trainables(r)) == 2
end

@testset "GRU" begin
    struct ModelGRU
        gru::GRU
        h0::AbstractVector
    end

    Flux.@layer :expand ModelGRU

    (m::ModelGRU)(x) = m.gru(m.h0, x)

    model = ModelGRU(GRU(2 => 4), zeros(Float32, 4))

    x = rand(Float32, 2, 3, 1)
    y = model(x)
    @test y isa Array{Float32, 3}
    @test size(y) == (4, 3, 1)
    test_gradients(model, x)

    # no initial state same as zero initial state
    gru = model.gru
    @test gru(x) ≈ gru(x, zeros(Float32, 4))
    
    # No Bias
    gru = GRU(2 => 4, bias=false)
    @test length(Flux.trainables(gru)) == 2
    test_gradients(gru, x)
end

@testset "GRUv3Cell" begin 
    r = GRUv3Cell(3 => 5)
    @test length(Flux.trainables(r)) == 4
    x = rand(Float32, 3)

    # Initial State is a single vector
    h = randn(Float32, 5)
    test_gradients(r, h, x)

    # no initial state same as zero initial state
    @test r(x) ≈ r(x, zeros(Float32, 5))

    # Now initial state has a batch dimension.
    h = randn(Float32, 5, 4)
    test_gradients(r, h, x)

    # The input sequence has no batch dimension.
    x = rand(Float32, 3)
    h = randn(Float32, 5)
    test_gradients(r, h, x)
end

@testset "GRUv3" begin
    struct ModelGRUv3
        gru::GRUv3
        h0::AbstractVector
    end

    Flux.@layer :expand ModelGRUv3

    (m::ModelGRUv3)(x) = m.gru(m.h0, x)

    model = ModelGRUv3(GRUv3(2 => 4), zeros(Float32, 4))

    x = rand(Float32, 2, 3, 1)
    y = model(x)
    @test y isa Array{Float32, 3}
    @test size(y) == (4, 3, 1)
    test_gradients(model, x)

    # no initial state same as zero initial state
    gru = model.gru
    @test gru(x) ≈ gru(x, zeros(Float32, 4))
end
