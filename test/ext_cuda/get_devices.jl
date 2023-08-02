cuda_device = Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["CUDA"]]

# should pass, whether or not CUDA is functional
@test typeof(cuda_device) <: Flux.FluxCUDADevice

if CUDA.functional()
  @test typeof(cuda_device.deviceID) <: CUDA.CuDevice 
else
  @test typeof(cuda_device.deviceID) <: Nothing
end

# testing get_device
if CUDA.functional()
  device = Flux.get_device()

  @test typeof(device) <: Flux.FluxCUDADevice
  @test typeof(device.deviceID) <: CUDA.CuDevice
  @test Flux._get_device_name(device) in Flux.supported_devices()

  # correctness of data transfer
  x = randn(5, 5)
  cx = x |> device
  @test cx isa CUDA.CuArray
  @test CUDA.device(cx).handle == device.deviceID.handle
end
