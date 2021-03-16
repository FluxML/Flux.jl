using BenchmarkTools
using Flux
using CUDA
using Zygote: pullback

function vgg16()
    Chain(
        Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(64),
        MaxPool((2,2)),
        Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(128),
        MaxPool((2,2)),
        Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256),
        MaxPool((2,2)),
        Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        MaxPool((2,2)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        MaxPool((2,2)),
        flatten,
        Dense(512, 4096, relu),
        Dropout(0.5),
        Dense(4096, 4096, relu),
        Dropout(0.5),
        Dense(4096, 10)
    )
end

fw(m, x) = m(x)
bw(back) = back(1f0)
fwbw(m, ps, x) = gradient(() -> sum(m(x)), ps)
  
function run_benchmark(; batchsize = 64, cuda=true)
    model = vgg16()
    x = rand(Float32, 32, 32, 3, batchsize)
    
    if cuda 
        model = model |> gpu
        x = x |> gpu
    end

    ps = Flux.params(model)
    y, back = pullback(() -> sum(model(x)), ps)


    if cuda
        CUDA.allowscalar(false)
        # CUDA.device!(3)
        println("  forward")
        fw(model, x); GC.gc(); CUDA.reclaim(); #warmup
        @btime CUDA.@sync(fw($model, $x)) teardown=(GC.gc(); CUDA.reclaim())

        println("  backward")
        bw(back); GC.gc(); CUDA.reclaim(); #warmup
        @btime CUDA.@sync(bw($back)) teardown=(GC.gc(); CUDA.reclaim())
        
        println("  forw and back")
        fwbw(model, ps, x); GC.gc(); CUDA.reclaim(); #warmup
        @btime CUDA.@sync(fwbw($model, $ps, $x)) teardown=(GC.gc(); CUDA.reclaim())
    else
        println("  forward")
        fw(model, x)  #warmup
        @btime fw($model, $x)

        println("  backward")
        bw(back)  #warmup
        @btime bw($back)

        println("  forw and back")
        fwbw(model, ps, x) # warmup
        @btime fwbw($model, $ps, $x)
    end
end

# RUN
println("CPU benchmark")
run_benchmark(cuda=false)
println("CUDA benchmark")
run_benchmark(cuda=true)
