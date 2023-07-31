x = rand(Float32, 10, 10)
if !(Flux.CUDA_LOADED[] || Flux.AMDGPU_LOADED[] || Flux.METAL_LOADED[])
    @test x === gpu(x)
end

@test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["CUDA"]]) <: Nothing
@test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["AMD"]]) <: Nothing
@test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["Metal"]]) <: Nothing
@test typeof(Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["CPU"]]) <: Flux.FluxCPUDevice

device = Flux.get_device()
@test typeof(device) <: Flux.FluxCPUDevice
@test device(x) == x
@test Flux._get_device_name(device) in Flux.supported_devices()
