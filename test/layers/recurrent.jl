
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

    Flux.@layer :expand ModelRNN

    (m::ModelRNN)(x) = m.rnn(x, m.h0)

    model = ModelRNN(RNN(2 => 4), zeros(Float32, 4))
    
    x = rand(Float32, 2, 3, 1)
    y = model(x)
    @test y isa Array{Float32, 3}
    @test size(y) == (4, 3, 1)
    test_gradients(model, x)

    x = rand(Float32, 2, 3)
    y = model(x)
    @test y isa Array{Float32, 2}
    @test size(y) == (4, 3)
    test_gradients(model, x)
end
