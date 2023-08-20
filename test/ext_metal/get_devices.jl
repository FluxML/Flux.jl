metal_device = Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["Metal"]]

# should pass, whether or not Metal is functional
@test typeof(metal_device) <: Flux.FluxMetalDevice

if Metal.functional()
    @test typeof(metal_device.deviceID) <: Metal.MTLDevice 
else
    @test typeof(metal_device.deviceID) <: Nothing
end

# testing get_device
if Metal.functional()
  metal_device = Flux.get_device()

  @test typeof(metal_device) <: Flux.FluxMetalDevice
  @test typeof(metal_device.deviceID) <: Metal.MTLDevice
  @test Flux._get_device_name(metal_device) in Flux.supported_devices()

  # correctness of data transfer
  x = randn(5, 5)
  cx = x |> metal_device
  @test cx isa Metal.MtlArray
  @test Metal.device(cx).registryID == metal_device.deviceID.registryID
end
