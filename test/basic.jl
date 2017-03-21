@net type TLP
  first
  second
  function (x)
    l1 = σ(first(x))
    l2 = softmax(second(l1))
  end
end

@net type Multi
  W
  V
  x -> (x*W, x*V)
end

Multi(in::Integer, out::Integer) =
  Multi(randn(in, out), randn(in, out))

@testset "Basics" begin

xs = randn(10)
d = Affine(10, 20)

@test d(xs) ≈ (xs'*d.W.x + d.b.x)[1,:]

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
    Flux.interpmodel(tlp, rand(10))
  catch e
    e
  end
  @test e.trace[end].func == :TLP
  @test e.trace[end-1].func == Symbol("Flux.Affine")
end

let m = Multi(10, 15)
  x = rand(10)
  @test all(isapprox.(m(x), (m.W.x' * x, m.V.x' * x)))
end

end
