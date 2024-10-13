amdgpu_device = gpu_device()

# should pass, whether or not AMDGPU is functional
@test typeof(amdgpu_device) <: Flux.AMDGPUDevice

# testing get_device
dense_model = Dense(2 => 3)     # initially lives on CPU
weight = copy(dense_model.weight)           # store the weight
bias = copy(dense_model.bias)               # store the bias

amdgpu_device = gpu_device()

@test typeof(amdgpu_device) <: Flux.AMDGPUDevice

# correctness of data transfer
x = randn(Float32, 5, 5)
cx = x |> amdgpu_device
@test cx isa AMDGPU.ROCArray

# moving models to specific AMDGPU devices
for id in 0:(length(AMDGPU.devices()) - 1)
  current_amdgpu_device = gpu_device(id+1)

  global dense_model = dense_model |> current_amdgpu_device
  @test dense_model.weight isa AMDGPU.ROCArray
  @test dense_model.bias isa AMDGPU.ROCArray
  @test AMDGPU.device_id(AMDGPU.device(dense_model.weight)) == id + 1
  @test AMDGPU.device_id(AMDGPU.device(dense_model.bias)) == id + 1
  @test isequal(Flux.cpu(dense_model.weight), weight)
  @test isequal(Flux.cpu(dense_model.bias), bias)
end
# finally move to CPU, and see if things work
cdev = cpu_device()
dense_model = cdev(dense_model)
@test dense_model.weight isa Matrix
@test dense_model.bias isa Vector
