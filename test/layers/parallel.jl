using Test, Random
using Flux
using Flux: mse, crossentropy, throttle, @epochs, softmax
using Statistics: mean
using Base.Iterators: partition

@testset "Parallel" begin

    @testset "one lonely `LSTM()` in a `Parallel` layer - should behave like a single `LSTM()`" begin
        
        data = collect(partition(rand(10, 7), 10))

        # non recurrent
        # m = Chain(Dense(10,10))

        # LSTM
        # m = Chain(LSTM(10,10))
        # m = Chain(LSTM(10,10), Dense(10,10))
        
        # Parallel map/reduce
        # FIXME: loss behaves oddly without a final Dense layer! Is tracking working for `Parallel`?
        # m = Chain(Parallel([LSTM(10,10)]))  # NOTE: compare to `Chain(LSTM(10,10))`
        # m = Chain(Parallel([LSTM(10,10)]))
        # m = Chain(Parallel([LSTM(10,10)]), Dense(10,10))
        # m = Chain(Parallel([LSTM(10,10),LSTM(10,10)]), Dense(20,10))
        
        # bidirectional LSTM
        # FIXME: loss behaves oddly without a final Dense layer! Is tracking working for `Parallel`?
        # m = Chain(BiLSTM(10,10))

        # m = Chain(BiLSTM(10,10), Dense(20,10))  # default: reduce=Flux.concat
        # m = Chain(BiLSTM(10,10, reduce=sum), Dense(10,10))
        # m = Chain(BiLSTM(10,10, reduce=Flux.mul), Dense(10,10))
        # m = Chain(BiLSTM(10,10, reduce=Flux.mean), Dense(10,10))

        # peephole LSTM
        # m = Chain(PLSTM(10,10))
        # m = Chain(PLSTM(10,10), Dense(10,10))
        # m = Chain(BiPLSTM(10,10), Dense(20,10))

        # @show m
        # @show params(m)

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

        Flux.reset_parallel!(m)
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