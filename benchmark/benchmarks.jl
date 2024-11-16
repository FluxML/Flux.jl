# We run the benchmarks using AirspeedVelocity.jl

# To run benchmarks locally, first install AirspeedVelocity.jl:
# julia> using Pkg; Pkg.add("AirspeedVelocity"); Pkg.build("AirspeedVelocity")
# and make sure .julia/bin is in your PATH.

# Then commit the changes and run:
# $ benchpkg Functors --rev=mybranch,master --bench-on=mybranch 


using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @benchmarkable, @btime, @benchmark, judge
using Flux
using Optimisers: AdamW
using LinearAlgebra: BLAS
using Statistics, Random

using CUDA

const SUITE = BenchmarkGroup()
const BENCHMARK_CPU_THREADS = Threads.nthreads()
BLAS.set_num_threads(BENCHMARK_CPU_THREADS)

function setup_train_mlp()
    d_in = 128
    d_out = 128
    batch_size = 128
    num_iters = 10
    device = gpu_device()
    
    model = Dense(d_in => d_out) |> device
    x = randn(Float32, d_in, batch_size) |> device
    y = randn(Float32, d_out, batch_size) |> device
    opt = Flux.setup(AdamW(1e-3), model)
    for iter in 1:num_iters
        ŷ = model(x)
        # g = gradient(m -> Flux.mse(m(x), y), model)[1]
        # Flux.update!(opt, model, g)
        @info iter
        # GC.gc(true)
        CUDA.pool_status()
    end
    CUDA.pool_status()
end

@time setup_train_mlp()


# for _ in 1:10
#     g = gradient(m -> Flux.mse(m(x), y), model)[1]
#     Flux.update!(opt, model, g)
# end
GC.gc(true)
CUDA.reclaim()
CUDA.pool_status()
