cuda_device = Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["CUDA"]]

# should pass, whether or not CUDA is functional
@test typeof(cuda_device) <: Flux.FluxCUDADevice

@test typeof(cuda_device.deviceID) <: CUDA.CuDevice 

# testing get_device
dense_model = Dense(2 => 3)                 # initially lives on CPU
weight = copy(dense_model.weight)           # store the weight
bias = copy(dense_model.bias)               # store the bias

cuda_device = Flux.get_device()

@test typeof(cuda_device) <: Flux.FluxCUDADevice
@test typeof(cuda_device.deviceID) <: CUDA.CuDevice
@test Flux._get_device_name(cuda_device) in Flux.supported_devices()

# correctness of data transfer
x = randn(5, 5)
cx = x |> cuda_device
@test cx isa CUDA.CuArray
@test CUDA.device(cx).handle == cuda_device.deviceID.handle

# moving models to specific NVIDIA devices
for ordinal in 0:(length(CUDA.devices()) - 1)
  current_cuda_device = Flux.get_device("CUDA", ordinal)
  @test typeof(current_cuda_device.deviceID) <: CUDA.CuDevice
  @test current_cuda_device.deviceID.handle == ordinal

  global dense_model = dense_model |> current_cuda_device
  @test dense_model.weight isa CUDA.CuArray
  @test dense_model.bias isa CUDA.CuArray
  @test CUDA.device(dense_model.weight).handle == ordinal
  @test CUDA.device(dense_model.bias).handle == ordinal
  @test isequal(Flux.cpu(dense_model.weight), weight)
  @test isequal(Flux.cpu(dense_model.bias), bias)
end
# finally move to CPU, and see if things work
cpu_device = Flux.get_device("CPU")
dense_model = cpu_device(dense_model)
@test dense_model.weight isa Matrix
@test dense_model.bias isa Vector
