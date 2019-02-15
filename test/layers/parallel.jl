using Test, Random
using Flux
using Flux: mse, crossentropy, throttle, @epochs, softmax
using Statistics: mean
using Base.Iterators: partition

@testset "Parallel" begin

    @testset "one lonely parallel layer - should behave like one LSTM()" begin
        
        data = collect(partition(rand(10, 7), 10))

        # m = Chain(Dense(10,10))
        # m = Chain(LSTM(10,10))
        # m = Chain(Parallel([LSTM(10,10)]))  # TODO: loss behaves oddly without final Dense layer?!
        m = Chain(Parallel([LSTM(10,10)]), Dense(10,10))
        # m = Chain(BiLSTM(10,10))            # TODO: loss behaves oddly without final Dense layer?!
        # m = Chain(BiLSTM(10,10), Dense(10,10)) # default: reduce=Flux.concat
        # m = Chain(BiLSTM(10,10, reduce=sum), Dense(10,10))
        # m = Chain(BiLSTM(10,10, reduce=Flux.mul), Dense(10,10))
        # m = Chain(BiLSTM(10,10, reduce=Flux.mean), Dense(10,10))

        before = Flux.data(m(data[1]))
        @test length(before) == 10

        function loss(x, y)
            l = mse(m(x), y)
            Flux.truncate_parallel!(m)
            l
        end

        function evalcb()
            error = mean(map(x -> loss(x, x), data))
            @show(error)
        end
        opt = ADAM()
        @epochs 3 Flux.train!(loss, params(m), zip(data, data), opt, cb = evalcb)

        after = Flux.data(m(data[1]))
        @test length(before) == length(after[:,end])
        @test before != after[:,end]

        Flux.reset_parallel!(m)
        after = Flux.data(m(data[1]))
        @test before != after
    end

    # @testset "reverse input for second layer" begin
    #     m = Parallel(layers, map = Dict{Int64,Function}(2 => reverse))
    # end

    # @testset "bidirectional layers with average merge" begin
    #     m = Parallel(layers, map = Dict{Int64,Function}(2 => reverse), reduce = mean)
    # end

end