using Flux
using BenchmarkTools
using Random 

function test_BN(shape)
    Random.seed!(17)
    x = rand(Float32, shape)
    bn = BatchNorm(shape[end-1])
    @btime $bn($x)
end

for shape in [  (10,10),
                (100,100),
                (1000,1000),]
    test_BN(shape)
end
