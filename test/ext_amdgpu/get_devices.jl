amd_device = Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["AMD"]]

# should pass, whether or not AMDGPU is functional
@test typeof(amd_device) <: Flux.FluxAMDDevice

if AMDGPU.functional()
    @test typeof(amd_device.deviceID) <: AMDGPU.HIPDevice 
else
    @test typeof(amd_device.deviceID) <: Nothing
end

if AMDGPU.functional() && AMDGPU.functional(:MIOpen)
  device = Flux.get_device()

  @test typeof(device) <: Flux.FluxAMDDevice
  @test typeof(device.deviceID) <: AMDGPU.HIPDevice
  @test Flux._get_device_name(device) in Flux.supported_devices()

  # correctness of data transfer
  x = randn(5, 5)
  cx = x |> device
  @test cx isa AMDGPU.ROCArray
  @test AMDGPU.device_id(AMDGPU.device(cx)) == AMDGPU.device_id(device.deviceID)

  # moving models to specific NVIDIA devices
  m = Dense(2 => 3)     # initially lives on CPU
  weight = copy(m.weight)           # store the weight
  bias = copy(m.bias)               # store the bias
  for ordinal in 0:(length(AMDGPU.devices()) - 1)
    device = Flux.get_device("AMD", ordinal)
    @test typeof(device.deviceID) <: AMDGPU.HIPDevice
    @test AMDGPU.device_id(device.deviceID) == ordinal

    m = m |> device
    @test m.weight isa AMDGPU.ROCArray
    @test m.bias isa AMDGPU.ROCArray
    @test ADMGPU.device_id(AMDGPU.device(m.weight)) == ordinal
    @test ADMGPU.device_id(AMDGPU.device(m.bias)) == ordinal
    @test isequal(Flux.cpu(m.weight), weight)
    @test isequal(Flux.cpu(m.bias), bias)
  end
  # finally move to CPU, and see if things work
  cpu_device = Flux.get_device("CPU")
  m = cpu_device(m)
  @test m.weight isa Matrix
  @test m.bias isa Vector

end
