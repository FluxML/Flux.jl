x = rand(Float32, 10, 10)
if !(Flux.CUDA_LOADED[] || Flux.AMDGPU_LOADED[] || Flux.METAL_LOADED[])
    @test x === gpu(x)
end

dev = Flux.get_device()
@test typeof(dev) <: Flux.FluxCPUDevice
@test dev(x) == x
@test Flux._get_device_name(dev) in Flux.supported_devices()

# specifically getting CPU device
dev = Flux.get_device("CPU")
@test typeof(dev) <: Flux.FluxCPUDevice
