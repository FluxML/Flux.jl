amd_device = Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["AMD"]]

# should pass, whether or not AMDGPU is functional
@test typeof(amd_device) <: Flux.FluxAMDDevice

if AMDGPU.functional()
    @test typeof(amd_device.deviceID) <: AMDGPU.HIPDevice 
else
    @test typeof(amd_device.deviceID) <: Nothing
end
