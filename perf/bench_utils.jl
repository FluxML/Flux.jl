using BenchmarkTools
using Flux
using CUDA
using Zygote: pullback


fw(m, x) = m(x)
bw(back) = back(1f0)
fwbw(m, ps, x) = gradient(() -> sum(m(x)), ps)
  
function run_benchmark(model, x; cuda=true)
    
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
