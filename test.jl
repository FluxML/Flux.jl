using BenchmarkTools, Flux
using Zygote: pullback

using LinearAlgebra
BLAS.set_num_threads(1)

function perf_test(n)
    r = rand(Float32, n, n) 
    d = Dense(n, n, relu)
    println("  FORW")
    @btime sum($d($r))
    println("  GRADIENT")
    @btime gradient(() -> sum($d($r)), $(Flux.params(d)))
    @btime gradient((d) -> sum(d($r)), $d)

    println("  PULLBACK")
    y, back =  pullback((d) -> sum(d(r)), d)
    @btime pullback((d) -> sum(d($r)), $d)
    @btime $back(1f0)
end

println("SMALL NET n=2")
perf_test(2)
println("MEDIUM NET n=20")
perf_test(20)
println("LARGE NET n=200")
perf_test(200)
println("VERY LARGE NET n=2000")
perf_test(2000)