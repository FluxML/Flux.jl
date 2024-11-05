x = rand(Float32, 10, 10)
if gpu_device() isa CPUDevice
    @test x === gpu(x)
end

dev = Flux.cpu_device()
@test typeof(dev) <: Flux.CPUDevice
@test dev(x) == x

