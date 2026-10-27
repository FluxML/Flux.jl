# Paced GC of `trainstep!` on a GPU (#2795), run for each GPU backend.
function gpu_trainstep_gc_testsuite(dev)
  @testset "trainstep! paces the GC on $(nameof(typeof(dev)))" begin
    T = Flux.Train
    model = Chain(Dense(64 => 256, relu), Dense(256 => 10)) |> dev
    opt = Flux.setup(Adam(), model)
    x, y = randn(Float32, 64, 128) |> dev, randn(Float32, 10, 128) |> dev
    loss(m, x, y) = Flux.mse(m(x), y)

    n = T.GC_COUNT[]
    for _ in 1:3
      Flux.trainstep!(loss, model, (x, y), opt)
    end
    @test T.GC_COUNT[] > n
    @test haskey(T.STEP_PACERS, objectid(T._first_param(model)))

    n = T.GC_COUNT[]
    for _ in 1:3
      Flux.trainstep_withgradient!(loss, model, (x, y), opt; gc_interval = 1)
    end
    @test T.GC_COUNT[] == n + 3
    Flux.trainstep!(loss, model, (x, y), opt; gc_interval = 0)
    @test T.GC_COUNT[] == n + 3
  end
end
