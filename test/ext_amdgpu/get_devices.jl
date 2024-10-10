amdgpu_device = gpu_device()

# should pass, whether or not AMDGPU is functional
@test typeof(amdgpu_device) <: Flux.AMDGPUDevice

@test typeof(amdgpu_device.deviceID) <: AMDGPU.HIPDevice 

# testing get_device
dense_model = Dense(2 => 3)     # initially lives on CPU
weight = copy(dense_model.weight)           # store the weight
bias = copy(dense_model.bias)               # store the bias

amdgpu_device = Flux.get_device()

@test typeof(amdgpu_device) <: Flux.FluxAMDGPUDevice
@test typeof(amdgpu_device.deviceID) <: AMDGPU.HIPDevice
@test Flux._get_device_name(amdgpu_device) in Flux.supported_devices()

# correctness of data transfer
x = randn(Float32, 5, 5)
cx = x |> amdgpu_device
@test cx isa AMDGPU.ROCArray
@test AMDGPU.device_id(AMDGPU.device(cx)) == AMDGPU.device_id(amdgpu_device.deviceID)

# moving models to specific NVIDIA devices
for id in 0:(length(AMDGPU.devices()) - 1)
  current_amdgpu_device = Flux.get_device("AMDGPU", id)
  @test typeof(current_amdgpu_device.deviceID) <: AMDGPU.HIPDevice
  @test AMDGPU.device_id(current_amdgpu_device.deviceID) == id + 1

  global dense_model = dense_model |> current_amdgpu_device
  @test dense_model.weight isa AMDGPU.ROCArray
  @test dense_model.bias isa AMDGPU.ROCArray
  @test AMDGPU.device_id(AMDGPU.device(dense_model.weight)) == id + 1
  @test AMDGPU.device_id(AMDGPU.device(dense_model.bias)) == id + 1
  @test isequal(Flux.cpu(dense_model.weight), weight)
  @test isequal(Flux.cpu(dense_model.bias), bias)
end
# finally move to CPU, and see if things work
cpu_device = Flux.get_device("CPU")
dense_model = cpu_device(dense_model)
@test dense_model.weight isa Matrix
@test dense_model.bias isa Vector
