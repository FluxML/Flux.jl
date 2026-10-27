include("../test_common/gpu_train.jl")
gpu_trainstep_gc_testsuite(gpu_device(force=true))

@testset "trainstep! frees each step's arrays" begin
  model = Chain(Dense(256 => 1024, relu), Dense(1024 => 10)) |> gpu
  opt = Flux.setup(Adam(), model)
  x, y = CUDA.randn(Float32, 256, 512), CUDA.randn(Float32, 10, 512)
  loss(m, x, y) = Flux.mse(m(x), y)
  Flux.trainstep!(loss, model, (x, y), opt)

  GC.gc(true)
  step = CUDA.@allocated Flux.trainstep!(loss, model, (x, y), opt; gc_interval = 0)
  GC.gc(true)
  used = CUDA.used_memory()
  for _ in 1:20
    Flux.trainstep!(loss, model, (x, y), opt; gc_interval = 1)
  end
  @test CUDA.used_memory() - used < 3 * step
end
