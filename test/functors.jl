x = rand(Float32, 10, 10)
if !(Flux.CUDA_LOADED[] || Flux.AMD_LOADED[] || Flux.METAL_LOADED[])
    @test x === gpu(x)
end
