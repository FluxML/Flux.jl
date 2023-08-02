metal_device = Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["Metal"]]

# should pass, whether or not Metal is functional
@test typeof(metal_device) <: Flux.FluxMetalDevice

if Metal.functional()
    @test typeof(metal_device.deviceID) <: Metal.MTLDevice 
else
    @test typeof(metal_device.deviceID) <: Nothing
end
