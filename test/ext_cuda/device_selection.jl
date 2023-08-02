cuda_device = Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["CUDA"]]

# should pass, whether or not CUDA is functional
@test typeof(cuda_device) <: Flux.FluxCUDADevice

if CUDA.functional()
    @test typeof(cuda_device.deviceID) <: CUDA.CuDevice 
else
    @test typeof(cuda_device.deviceID) <: Nothing
end
