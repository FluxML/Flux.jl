using MXNet
Flux.loadmx()

@testset "MXNet" begin

xs, ys = rand(1, 20), rand(1, 20)
d = Affine(20, 10)

dm = mxnet(d)
@test d(xs) ≈ dm(xs)

m = Multi(20, 15)
mm = mxnet(m)
@test all(isapprox.(mm(xs, ys), m(xs, ys)))

@testset "Tuple I/O" begin
  @test mxnet(@net x -> (x,))([1,2,3]) == ([1,2,3],)
  @test mxnet(@net x -> x[1].*x[2])(([1,2,3],[4,5,6])) == [4,10,18]
end

@testset "Recurrence" begin
  seq = batchone(Seq(rand(10) for i = 1:3))
  r = unroll(Recurrent(10, 5), 3)
  rm = mxnet(r)
  @test r(seq) ≈ rm(seq)
end

@testset "Backward Pass" begin
  d′ = deepcopy(d)
  @test dm(xs) ≈ d(xs)
  @test dm(xs) ≈ d′(xs)

  Δ = back!(dm, randn(1, 10), xs)
  @test length(Δ[1]) == 20
  update!(dm, 0.1)

  @test dm(xs) ≈ d(xs)
  @test dm(xs) ≉ d′(xs)
end

@testset "Native interface" begin
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
  dm = mxnet(model)
  e = try dm(rand(1, 10))
  catch e e end

  @test isa(e, DataFlow.Interpreter.Exception)
  @test e.trace[1].func == Symbol("Flux.Affine")
  @test e.trace[2].func == :TLP
end

end
