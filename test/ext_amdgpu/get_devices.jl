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
end
