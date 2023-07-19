x = rand(Float32, 10, 10)
if Flux.CUDA_LOADED[]
    @test x !== gpu(x)
elseif Flux.AMD_LOADED[]
    @test x !== gpu(x)
elseif Flux.METAL_LOADED[]
    @test x !== gpu(x)
else
    @test x === gpu(x)
end
