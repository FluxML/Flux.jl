x = rand(Float32, 10, 10)
if !(Flux.CUDA_LOADED[] || Flux.AMDGPU_LOADED[] || Flux.METAL_LOADED[])
    @test x === gpu(x)
end

dev = Flux.cpu_device()
@test typeof(dev) <: Flux.CPUDevice
@test dev(x) == x

