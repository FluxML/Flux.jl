
@testset "RNNCell" begin
    function loss1(r, x, h)
        for x_t in x
            h = r(x_t, h)
        end
        return mean(h.^2)
    end

    function loss2(r, x, h)
        y = [r(x_t, h) for x_t in x]
        return sum(mean, y)
    end

    function loss3(r, x, h)
        y = []
        for x_t in x
            h = r(x_t, h)
            y = [y..., h]
        end
        return sum(mean, y)
    end

    function loss4(r, x, h)
        y = []
        for x_t in x
            h = r(x_t, h)
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
    test_gradients(r, x, h, loss=loss1) # for loop
    test_gradients(r, x, h, loss=loss2) # comprehension
    test_gradients(r, x, h, loss=loss3) # splat
    test_gradients(r, x, h, loss=loss4) # vcat and stack

    # initial states are zero
    @test Flux.initialstates(r) ≈ zeros(Float32, 5)

    # no initial state same as zero initial state
    @test r(x[1]) ≈ r(x[1], zeros(Float32, 5))

    # Now initial state has a batch dimension.
    h = randn(Float32, 5, 4)
    test_gradients(r, x, h, loss=loss4)

    # The input sequence has no batch dimension.
    x = [rand(Float32, 3) for _ in 1:6]
    h = randn(Float32, 5)
    test_gradients(r, x, h, loss=loss4)

    
    # No Bias 
    r = RNNCell(3 => 5, bias=false)
    @test length(Flux.trainables(r)) == 2
    test_gradients(r, x, h, loss=loss4)
end

@testset "RNN" begin
    struct ModelRNN
        rnn::RNN
        h0::AbstractVector
    end

    Flux.@layer ModelRNN

    (m::ModelRNN)(x) = m.rnn(x, m.h0)

    model = ModelRNN(RNN(2 => 4), zeros(Float32, 4))
    
    x = rand(Float32, 2, 3, 1)
    y = model(x)
    @test y isa Array{Float32, 3}
    @test size(y) == (4, 3, 1)
    test_gradients(model, x)

    rnn = model.rnn
    # initial states are zero
    @test Flux.initialstates(rnn) ≈ zeros(Float32, 4)
    
    # no initial state same as zero initial state
    @test rnn(x) ≈ rnn(x, zeros(Float32, 4))

    x = rand(Float32, 2, 3)
    y = model(x)
    @test y isa Array{Float32, 2}
    @test size(y) == (4, 3)
    test_gradients(model, x)
end

@testset "LSTMCell" begin

    function loss(r, x, hc)
        h, c = hc
        h′ = []
        c′ = []
        for x_t in x
            h, c = r(x_t, (h, c))
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
    test_gradients(cell, x[1], (h, c), loss = (m, x, hc) -> mean(m(x, hc)[1]))
    test_gradients(cell, x, (h, c), loss = loss)

    # initial states are zero
    h0, c0 = Flux.initialstates(cell)
    @test h0 ≈ zeros(Float32, 5)
    @test c0 ≈ zeros(Float32, 5)

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

    Flux.@layer ModelLSTM

    (m::ModelLSTM)(x) = m.lstm(x, (m.h0, m.c0))

    model = ModelLSTM(LSTM(2 => 4), zeros(Float32, 4), zeros(Float32, 4))
    
    x = rand(Float32, 2, 3, 1)
    h = model(x)
    @test h isa Array{Float32, 3}
    @test size(h) == (4, 3, 1)
    test_gradients(model, x)

    x = rand(Float32, 2, 3)
    h = model(x)
    @test h isa Array{Float32, 2}
    @test size(h) == (4, 3)
    test_gradients(model, x, loss = (m, x) -> mean(m(x)[1]))

    # test default initial states
    lstm = model.lstm
    h = lstm(x)
    @test h isa Array{Float32, 2}
    @test size(h) == (4, 3)
    
    # initial states are zero
    h0, c0 = Flux.initialstates(lstm)
    @test h0 ≈ zeros(Float32, 4)
    @test c0 ≈ zeros(Float32, 4)

    # no initial state same as zero initial state
    h1 = lstm(x, (zeros(Float32, 4), zeros(Float32, 4)))
    @test h ≈ h1
end

@testset "GRUCell" begin
    function loss(r, x, h)
        y = []
        for x_t in x
            h = r(x_t, h)
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
    test_gradients(r, x, h; loss)

    # initial states are zero
    @test Flux.initialstates(r) ≈ zeros(Float32, 5)

    # no initial state same as zero initial state
    @test r(x[1]) ≈ r(x[1], zeros(Float32, 5))

    # Now initial state has a batch dimension.
    h = randn(Float32, 5, 4)
    test_gradients(r, x, h; loss)

    # The input sequence has no batch dimension.
    x = [rand(Float32, 3) for _ in 1:6]
    h = randn(Float32, 5)
    test_gradients(r, x, h; loss)

    # No Bias 
    r = GRUCell(3 => 5, bias=false)
    @test length(Flux.trainables(r)) == 2
end

@testset "GRU" begin
    struct ModelGRU
        gru::GRU
        h0::AbstractVector
    end

    Flux.@layer ModelGRU

    (m::ModelGRU)(x) = m.gru(x, m.h0)

    model = ModelGRU(GRU(2 => 4), zeros(Float32, 4))

    x = rand(Float32, 2, 3, 1)
    y = model(x)
    @test y isa Array{Float32, 3}
    @test size(y) == (4, 3, 1)
    test_gradients(model, x)

    
    gru = model.gru
    # initial states are zero
    @test Flux.initialstates(gru) ≈ zeros(Float32, 4)

    # no initial state same as zero initial state
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
    test_gradients(r, x, h)

    # initial states are zero
    @test Flux.initialstates(r) ≈ zeros(Float32, 5)

    # no initial state same as zero initial state
    @test r(x) ≈ r(x, zeros(Float32, 5))

    # Now initial state has a batch dimension.
    h = randn(Float32, 5, 4)
    test_gradients(r, x, h)

    # The input sequence has no batch dimension.
    x = rand(Float32, 3)
    h = randn(Float32, 5)
    test_gradients(r, x, h)
end

@testset "GRUv3" begin
    struct ModelGRUv3
        gru::GRUv3
        h0::AbstractVector
    end

    Flux.@layer ModelGRUv3

    (m::ModelGRUv3)(x) = m.gru(x, m.h0)

    model = ModelGRUv3(GRUv3(2 => 4), zeros(Float32, 4))

    x = rand(Float32, 2, 3, 1)
    y = model(x)
    @test y isa Array{Float32, 3}
    @test size(y) == (4, 3, 1)
    test_gradients(model, x)

    gru = model.gru
    # initial states are zero
    @test Flux.initialstates(gru) ≈ zeros(Float32, 4)
    
    # no initial state same as zero initial state
    @test gru(x) ≈ gru(x, zeros(Float32, 4))
end

@testset "Recurrence" begin
    for rnn in [RNN(2 => 3), LSTM(2 => 3), GRU(2 => 3)]
        cell = rnn.cell
        rec = Recurrence(cell)
        x = rand(Float32, 2, 3, 4)
        @test rec(x) ≈ rnn(x)
    end
end
