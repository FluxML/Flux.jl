@testset "Basics" begin

xs = randn(1, 10)
d = Affine(10, 20)

@test d(xs) ≈ (xs*d.W.x + d.b.x)

d1 = @net x -> x * d.W + d.b

@test d(xs) == d1(xs)

let
  # In 0.6 `.+` evaluates to an anon function, so we must match on that.
  @capture(syntax(d), _Frame(_Line(bplus_(x_[1] * W_, b_))))
  @test isa(x, DataFlow.Input) && isa(W, Param) && isa(b, Param)
end

let a1 = Affine(10, 20), a2 = Affine(20, 15)
  tlp = TLP(a1, a2)
  @test tlp(xs) ≈ softmax(a2(σ(a1(xs))))
  @test Flux.interpmodel(tlp, xs) ≈ softmax(a2(σ(a1(xs))))
  @test Flux.infer(tlp, (1, 10)) == (1,15)
end

let tlp = TLP(Affine(10, 21), Affine(20, 15))
  e = try
    Flux.interpmodel(tlp, rand(1, 10))
  catch e
    e
  end
  @test e.trace[end].func == :TLP
  @test e.trace[end-1].func == Symbol("Flux.Affine")
end

let m = Multi(10, 15)
  x, y = rand(1, 10), rand(1, 10)
  @test all(isapprox.(m(x, y), (x * m.W.x, y * m.V.x)))
  @test all(isapprox.(m(x, y), Flux.interpmodel(m, x, y)))
end

end
