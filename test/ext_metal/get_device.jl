device = Flux.get_device()

@test typeof(device) <: Flux.FluxMetalDevice
@test typeof(device.deviceID) <: Metal.MTLDevice
@test Flux._get_device_name(device) in Flux.supported_devices()

# correctness of data transfer
x = randn(5, 5)
cx = x |> device
@test cx isa Metal.MtlArray
@test Metal.device(cx).registryID == device.deviceID.registryID
