xs = randn(10)

d = Affine(10, 20)

@test d(xs) ≈ (xs'*d.W.x + d.b.x)[1,:]

let
  @capture(syntax(d), _Frame(_Line(x_[1] * W_ + b_)))
  @test isa(x, Input) && isa(W, Param) && isa(b, Param)
end

@net type TLP
  first
  second
  function (x)
    l1 = σ(first(x))
    l2 = softmax(second(l1))
  end
end

let a1 = Affine(10, 20), a2 = Affine(20, 15)
  tlp = TLP(a1, a2)
  @test tlp(xs) ≈ softmax(a2(σ(a1(xs))))
end
