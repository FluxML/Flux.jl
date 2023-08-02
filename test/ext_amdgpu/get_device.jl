device = Flux.get_device()

@test typeof(device) <: Flux.FluxAMDDevice
@test typeof(device.deviceID) <: AMDGPU.HIPDevice
@test Flux._get_device_name(device) in Flux.supported_devices()

# correctness of data transfer
x = randn(5, 5)
cx = x |> device
@test cx isa AMDGPU.ROCArray
@test AMDGPU.device_id(AMDGPU.device(cx)) == AMDGPU.device_id(device.deviceID)
