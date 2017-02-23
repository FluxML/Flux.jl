@net type TLP
  first
  second
  function (x)
    l1 = σ(first(x))
    l2 = softmax(second(l1))
  end
end

@testset "Basics" begin

xs = randn(10)
d = Affine(10, 20)

@test d(xs) ≈ (xs'*d.W.x + d.b.x)[1,:]

let
  @capture(syntax(d), _Frame(_Line(x_[1] * W_ + b_)))
  @test isa(x, DataFlow.Input) && isa(W, Param) && isa(b, Param)
end

let a1 = Affine(10, 20), a2 = Affine(20, 15)
  tlp = TLP(a1, a2)
  @test tlp(xs) ≈ softmax(a2(σ(a1(xs))))
  @test Flux.infer(tlp, (1, 10)) == (1,15)
end

let tlp = TLP(Affine(10, 21), Affine(20, 15))
  e = try
    tlp(rand(10))
  catch e
    e
  end
  @test e.trace[end].func == :TLP
  @test e.trace[end-1].func == Symbol("Flux.Affine")
end

end
