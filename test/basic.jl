xs = randn(10)' # TODO: batching semantics

d = Affine(10, 20)

@test d(xs) == xs*d.W.x + d.b.x

@testset begin
  @capture(syntax(d), x_[1] * W_ + b_)
  @test isa(x, Input)
  @test isa(W, Param)
  @test isa(b, Param)
end
