using MXNet
Flux.loadmx()

@testset "MXNet" begin

xs, ys = rand(1, 20), rand(1, 20)
d = Affine(20, 10)

dm = mxnet(d)
@test d(xs) ≈ dm(xs)

test_tupleio(mxnet)
test_recurrence(mxnet)
test_stacktrace(mxnet)
test_back(mxnet)
test_anon(mxnet)

using Flux: MaxPool

@testset "Native interface" begin
  f = Flux.MX.FeedForward(Chain(d, softmax))
  @test mx.infer_shape(f.arch, data = (20, 1))[2] == [(10, 1)]

  m = Chain(Input(28,28), Conv2D((5,5), out = 3), MaxPool((2,2)),
            flatten, Affine(1587, 10), softmax)
  f = Flux.MX.FeedForward(m)
  # TODO: test run
  @test mx.infer_shape(f.arch, data = (20, 20, 5, 1))[2] == [(10, 1)]
end

@testset "Duplicate parameters" begin
  a = Affine(10, 10)
  d = Chain(a, a)
  m = mxnet(d)
  m(randn(1, 10))
  @test length(m.graph.params) == 2
end

end
