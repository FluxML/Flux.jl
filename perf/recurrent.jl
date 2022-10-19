
struct RNNWrapper{T}
    rnn::T
end
Flux.@functor RNNWrapper

# Need to specialize for RNNWrapper.
fw(r::RNNWrapper, X::Vector{<:AbstractArray}) = begin
    Flux.reset!(r.rnn)
    [r.rnn(x) for x in X]
end

fw(r::RNNWrapper, X) = begin
    Flux.reset!(r.rnn)
    r.rnn(X)
end

fwbw(r::RNNWrapper, ps, X::Vector{<:AbstractArray}) =
    gradient(ps) do
        y = fw(r, X)
        return sum(sum(y))
    end

pb(r::RNNWrapper, ps, X::Vector{<:AbstractArray}) =
    pullback(ps) do
        y = fw(r, X)
        return sum(sum(y))
    end

function rnn_benchmark_sweep(data_creator::Function, rnn_type)
    for n in [2, 20, 200, 1000], ts in [1, 4, 16, 64]
        x, x_n = data_creator(n, ts)
        model = RNNWrapper(rnn_type(n, n))

        println("$rnn_type $x_n CPU n=$n, ts=$ts")
        run_benchmark(model, x, cuda = false)

        println("$rnn_type $x_n CUDA n=$n, ts=$ts")
        try
            run_benchmark(model, x, cuda = true)
        catch ex
            @show typeof(ex)
            if ex isa OutOfGPUMemoryError
                @warn "Not enough GPU memory to run test"
            else
                rethrow(ex)
            end
        end
    end
end

for rnn_type in [Flux.RNN, Flux.GRU, Flux.LSTM]
    rnn_benchmark_sweep(rnn_type) do n, ts
        return [randn(Float32, n, n) for _ in 1:ts], "Vec"
    end
end

for rnn_type in [Flux.RNN, Flux.GRU, Flux.LSTM]
    rnn_benchmark_sweep(rnn_type) do n, ts
        return randn(Float32, n, n, ts), "Block"
    end
end
