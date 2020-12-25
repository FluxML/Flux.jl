using Flux
using BenchmarkTools
using Random 
using Zygote: pullback

function test_BN(shape)
    Random.seed!(17)
    println("# Shape $shape")
    x = rand(Float32, shape)
    bn = BatchNorm(shape[end-1])
    # println("### forward")
    # @btime $bn($x)
    println("forward in gradient context")
    @btime pullback(x -> sum(sin.($bn(x))), $x)
    y, back = pullback(x -> sum(sin.(bn(x))), x)
    println("pullback")
    @btime $back(1f0)
end

for shape in [  (10,10),
                (100,100),
                (1000,1000),]
    test_BN(shape)
end