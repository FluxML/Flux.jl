amd_device = Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["AMD"]]

# should pass, whether or not AMDGPU is functional
@test typeof(amd_device) <: Flux.FluxAMDDevice

@test typeof(amd_device.deviceID) <: AMDGPU.HIPDevice 

# testing get_device
dense_model = Dense(2 => 3)     # initially lives on CPU
weight = copy(dense_model.weight)           # store the weight
bias = copy(dense_model.bias)               # store the bias

amd_device = Flux.get_device()

@test typeof(amd_device) <: Flux.FluxAMDDevice
@test typeof(amd_device.deviceID) <: AMDGPU.HIPDevice
@test Flux._get_device_name(amd_device) in Flux.supported_devices()

# correctness of data transfer
x = randn(5, 5)
cx = x |> amd_device
@test cx isa AMDGPU.ROCArray
@test AMDGPU.device_id(AMDGPU.device(cx)) == AMDGPU.device_id(amd_device.deviceID)

# moving models to specific NVIDIA devices
for ordinal in 0:(length(AMDGPU.devices()) - 1)
  current_amd_device = Flux.get_device("AMD", ordinal)
  @test typeof(current_amd_device.deviceID) <: AMDGPU.HIPDevice
  @test AMDGPU.device_id(current_amd_device.deviceID) == ordinal + 1

  global dense_model = dense_model |> current_amd_device
  @test dense_model.weight isa AMDGPU.ROCArray
  @test dense_model.bias isa AMDGPU.ROCArray
  @test AMDGPU.device_id(AMDGPU.device(dense_model.weight)) == ordinal + 1
  @test AMDGPU.device_id(AMDGPU.device(dense_model.bias)) == ordinal + 1
  @test isequal(Flux.cpu(dense_model.weight), weight)
  @test isequal(Flux.cpu(dense_model.bias), bias)
end
# finally move to CPU, and see if things work
cpu_device = Flux.get_device("CPU")
dense_model = cpu_device(dense_model)
@test dense_model.weight isa Matrix
@test dense_model.bias isa Vector
