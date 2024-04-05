
@testset "get_device()" begin
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

@testset "get_device(Metal)" begin
  metal_device = Flux.get_device("Metal")

  @test typeof(metal_device) <: Flux.FluxMetalDevice
  @test typeof(metal_device.deviceID) <: Metal.MTLDevice
  @test Flux._get_device_name(metal_device) in Flux.supported_devices()

  metal_device = Flux.get_device("Metal", 0)

  @test typeof(metal_device) <: Flux.FluxMetalDevice
  @test typeof(metal_device.deviceID) <: Metal.MTLDevice
  @test Flux._get_device_name(metal_device) in Flux.supported_devices()
end

