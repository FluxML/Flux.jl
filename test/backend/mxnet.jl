using MXNet
Flux.loadmx()

@testset "MXNet" begin

xs = rand(20)
d = Affine(20, 10)

dm = mxnet(d, (1, 20))
@test d(xs) ≈ dm(xs)

m = Multi(20, 15)
mm = mxnet(m, (1, 20))
@test all(isapprox.(mm(xs), m(xs)))

@testset "Backward Pass" begin
  d′ = deepcopy(d)
  @test dm(xs) ≈ d(xs)
  @test dm(xs) ≈ d′(xs)

  Δ = back!(dm, randn(10), xs)
  @test length(Δ) == 20
  update!(dm, 0.1)

  @test dm(xs) ≈ d(xs)
  @test dm(xs) ≉ d′(xs)
end

@testset "FeedForward interface" begin
  f = mx.FeedForward(Chain(d, softmax))
  @test mx.infer_shape(f.arch, data = (20, 1))[2] == [(10, 1)]

  m = Chain(Input(28,28), Conv2D((5,5), out = 3), MaxPool((2,2)),
            flatten, Affine(1587, 10), softmax)
  f = mx.FeedForward(m)
  # TODO: test run
  @test mx.infer_shape(f.arch, data = (20, 20, 5, 1))[2] == [(10, 1)]
end

@testset "Stack Traces" begin
  model = TLP(Affine(10, 20), Affine(21, 15))
  info("The following warning is normal")
  e = try mxnet(model, (10, 1))
  catch e e end

  @test isa(e, DataFlow.Interpreter.Exception)
  @test e.trace[1].func == Symbol("Flux.Affine")
  @test e.trace[2].func == :TLP
end

end
