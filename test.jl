using LoopVectorization, Zygote
using BenchmarkTools, Random
using Flux: relu, gpu
using CUDA, KernelAbstractions
using Tullio

linear(W, x, b) = W*x .+ b

function sigm(x)
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(one(t) + t), t / (one(t) + t))
end

for layer in (:Dense1, :Dense2, :Dense3, :Dense4, :Dense5, :Dense6, :Dense7, :Dense8)
    @eval struct $layer{S<:AbstractArray, T<:AbstractArray, F}
        W::S
        b::T
        σ::F
    end
end
  
function (a::Dense1)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    σ.(W*x .+ b)
end

function (a::Dense2)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    @avx σ.(W*x .+ b)
end
  
function (a::Dense3)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    σ.(linear(W, x, b))
end

function (a::Dense4)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    @avx σ.(linear(W, x, b))
end

function (a::Dense5)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    map(σ, linear(W, x, b))
end

function (a::Dense6)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    vmap(σ, linear(W, x, b))
end

function (a::Dense7)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    y = linear(W, b, σ)
    @tullio z[i,k] := σ.(y[i,k])
end

function (a::Dense8)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    @tullio y[i,k] := W[i,j]*x[j,k]
    @tullio z[i,k] := σ.(y[i,k])
end

function test()
    n = 500
    for batchsize in (100,)
        println("\n@@@ batchsize = $batchsize")
        x = randn(Float32, n, batchsize)
        W, b = randn(Float32, n, n)./√n, randn(Float32, n)
        cu_x, cu_W, cu_b = x |> gpu, W |> gpu, b |> gpu

        # for act in (identity, relu, exp)
        for act in (exp, sigm)
            println("\n@@ ACTIVATION $act")
            println(" broadcast")
            @btime $act.($x);
            println(" grad broadcast ")
            @btime gradient(x -> sum($act.(x)), $x);
            println(" @avx broadcast")
            @btime @avx $act.($x);
            println(" grad @avx broadcast")
            try; @btime gradient(x -> sum(@avx $act.(x)), $x); catch; println("ERROR!"); end
            println(" @tullio ")
            @btime @tullio y[i,j] := $act($x[i,j]);
            println(" grad @tullio")
            gradient(x -> (@tullio s := act(x[i,j])), x)
            try; @btime gradient(x -> (@tullio s := $act(x[i,j])), $x); catch; println("ERROR!"); end
            println(" map")
            @btime map($act, $x);
            println(" grad map")
            try; @btime gradient(x -> sum(map($act, x)), $x); catch; println("ERROR!"); end
            println(" vmap")
            @btime vmap($act, $x);
            println(" grad vmap")
            try; @btime gradient(x -> sum(vmap($act, x)), $x); catch; println("ERROR!"); end
            
            println(" GPU broadcast")
            @btime CUDA.@sync $act.($cu_x);
            println(" GPU @avx broadcast")
            try; @btime @avx $act.($cu_x); catch; println("ERROR!"); end
            println(" GPU @tullio ")
            @btime @tullio y[i,j] := $act.($cu_x[i,j]);
            println(" GPU grad @tullio")
            try; @btime gradient(x -> (@tullio s := $act(x[i,j])), $cu_x); catch; println("ERROR!"); end
            
            println(" GPU map")
            @btime map($act, $cu_x);
            println(" GPU vmap")
            try; @btime vmap($act, $cu_x); catch; println("ERROR!"); end
    
            for layer in (Dense1, Dense2, Dense3, Dense4, Dense5, Dense6, Dense7, Dense8)
                println("\n LAYER $layer")
                m = layer(W, b, act)
                @btime $m($x);
                println(" grad")
                try; @btime gradient(x-> sum($m(x)), $x); catch; println("ERROR!"); end
                
                println(" GPU LAYER $layer")
                m = layer(cu_W, cu_b, act)
                try; @btime $m($cu_x); catch; println("ERROR!"); end
                println(" grad")
                try; @btime gradient(x-> sum($m(x)), $cu_x); catch; println("ERROR!"); end    
            end          
        end
    end
end