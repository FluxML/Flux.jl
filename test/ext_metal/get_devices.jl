@testset "get_device()" begin
  metal_device = Flux.get_device()

  @test typeof(metal_device) <: Flux.MetalDevice

  # correctness of data transfer
  x = randn(5, 5)
  cx = x |> metal_device
  @test cx isa Metal.MtlArray
  @test Metal.device(cx).registryID == metal_device.deviceID.registryID
end

@testset "gpu_device()" begin
  metal_device = gpu_device()

  @test typeof(metal_device) <: Flux.MetalDevice

  metal_device = gpu_device(0)

  @test typeof(metal_device) <: Flux.MetalDevice
end

