using MXNet
Flux.loadmx()

@testset "MXNet" begin

xs = rand(20)
d = Affine(20, 10)

dm = mxnet(d, (20, 1))
@test d(xs) ≈ dm(xs)

@testset "FeedForward interface" begin
  # TODO: test run
  f = mx.FeedForward(Chain(d, softmax))
  @test mx.infer_shape(f.arch, data = (20, 1))[2] == [(10, 1)]

  m = Chain(Input(28,28), Conv2D((5,5), out = 3), MaxPool((2,2)),
            flatten, Affine(1587, 10), softmax)
  f = mx.FeedForward(m)
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
