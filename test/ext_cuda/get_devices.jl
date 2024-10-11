cuda_device = gpu_device()

# should pass, whether or not CUDA is functional
@test typeof(cuda_device) <: Flux.CUDADevice

# testing get_device
dense_model = Dense(2 => 3)                 # initially lives on CPU
weight = copy(dense_model.weight)           # store the weight
bias = copy(dense_model.bias)               # store the bias

cuda_device = Flux.get_device()

@test typeof(cuda_device) <: Flux.CUDADevice

# correctness of data transfer
x = randn(5, 5)
cx = x |> cuda_device
@test cx isa CUDA.CuArray

# moving models to specific NVIDIA devices
for id in 0:(length(CUDA.devices()) - 1)
  current_cuda_device = gpu_device(id+1)
  @test typeof(current_cuda_device) <: Flux.CUDADevice

  global dense_model = dense_model |> current_cuda_device
  @test dense_model.weight isa CUDA.CuArray
  @test dense_model.bias isa CUDA.CuArray
  @test CUDA.device(dense_model.weight).handle == id
  @test CUDA.device(dense_model.bias).handle == id
  @test isequal(Flux.cpu(dense_model.weight), weight)
  @test isequal(Flux.cpu(dense_model.bias), bias)
end
# finally move to CPU, and see if things work
cdev = cpu_device()
dense_model = cdev(dense_model)
@test dense_model.weight isa Matrix
@test dense_model.bias isa Vector
