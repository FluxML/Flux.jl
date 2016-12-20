xs = randn(10)' # TODO: batching semantics

d = Affine(10, 20)

@test d(xs) == xs*d.W.x + d.b.x

let
  @capture(syntax(d), _Line(x_[1] * W_ + b_))
  @test isa(x, Input) && isa(W, Param) && isa(b, Param)
end
