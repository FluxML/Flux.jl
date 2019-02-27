using Test, Random
using Flux
using Flux: @epochs
using Statistics: mean
using Base.Iterators: partition
# using CuArrays

@testset "Parallel" begin

    data = gpu.(collect(partition(rand(10, 7), 10)))

    models = [
        # non recurrent layers
        Chain(Parallel([Dense(10,10), Dense(10,10)]), Dense(20,10)),

        # recurrent layers - reduce defaults to `Flux.concat: 10 -> 10
        Parallel([LSTM(10,10)]),
        Chain(Parallel([LSTM(10,10)])),

        # for reduce see: `sum`, `mean`, `Flux.mul`, `Flux.concat`
        Parallel([LSTM(10,5), LSTM(10,5)]),
        Parallel([LSTM(10,10), LSTM(10,10)], reduce=sum),
        Chain(Parallel([LSTM(10,10), LSTM(10,10)], reduce=mean)),

        # reduce can be `Flux.concat` -> reduction is effectifly done by a Dense layer
        Chain(Parallel([LSTM(10,10)]), Dense(10,10)),
        Chain(Parallel([LSTM(10,10), LSTM(10,10)]), Dense(20,10)),

        # bidirectional LSTM
        Parallel([LSTM(10,10), LSTM(10,10)],
            map = Dict{Int64,Function}(2 => reverse),
            inv = Dict{Int64,Function}(2 => reverse),
            reduce = sum),

        # BiLSTM - a convenience layer, which makes use of `Parallel` and the MapReduce approach
        # for reduce see also: `sum`, `mean`, `Flux.mul`, `Flux.concat`
        Bi(LSTM(10, 10), sum),
        Chain(BiLSTM(10,10, sum)),

        # peephole LSTM - an modified LSTM layer commonly used in image processing.
        Chain(PLSTM(10,10)),
        Chain(Parallel([PLSTM(10,10), PLSTM(10,10)], reduce=sum)),

        # BiPLSTM - a convenience layer, which makes use of `Parallel` and the MapReduce approach
        Chain(BiPLSTM(10,10), Dense(20,10)),
        Chain(BiPLSTM(10,10), BiPLSTM(20,10), Dense(20,10)),
    ]

    @testset "models using a `Parallel` layer" for (i,m) in enumerate(models)
        println("\n\ntest ($i)\n")
        @show m
        sleep(0.1)

        gpu(m)

        before = Flux.data(m(data[1]))
        @test length(before) == 10 || length(before) == 20

        function loss(x, y)
            l = Flux.mse(m(x), y)
            Flux.truncate!(m)
            l
        end

        function evalcb()
            error = mean(map(x -> loss(x, x), data))
            @show(error)
        end
        opt = ADAM()
        @epochs 3 Flux.train!(loss, params(m), zip(data, data), opt, cb = evalcb)

        Flux.reset!(m)
        after = Flux.data(m(data[1]))
        @test length(before) == length(after[:,end]) || length(before) == 2 * length(after[:,end])
        @test before != after[:,end]

        Flux.reset!(m)
        after = Flux.data(m(data[1]))
        @test before != after
    end

end
