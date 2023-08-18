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


  # moving models to specific NVIDIA devices
  m = Dense(2 => 3)     # initially lives on CPU
  for ordinal in 0:(length(CUDA.devices()) - 1)
    device = Flux.get_device("CUDA", ordinal)
    @test typeof(device.deviceID) <: CUDA.CuDevice
    @test device.deviceID.handle == ordinal

    m = m |> device
    @test m.weight isa CUDA.CuArray
    @test m.bias isa CUDA.CuArray
    @test CUDA.device(m.weight).handle == ordinal
    @test CUDA.device(m.bias).handle == ordinal
  end
  # finally move to CPU, and see if things work
  cpu_device = Flux.get_device("CPU")
  m = cpu_device(m)
  @test m.weight isa Matrix
  @test m.bias isa Vector

end
