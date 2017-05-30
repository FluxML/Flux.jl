syntax(v::Vertex) = prettify(DataFlow.syntax(v))
syntax(x) = syntax(graph(x))

@testset "Basics" begin

xs = randn(1, 10)
d = Affine(10, 20)

@test d(xs) ≈ (xs*d.W.x + d.b.x)

d1 = @net x -> x * d.W + d.b

Flux.infer(d, (1, 10))

# Skip this before new DataFlow is released.
# let
#   @test @capture(syntax(d), _Frame(_Line((+).(x_[1] * W_, b_))))
#   @test isa(x, DataFlow.Input) && isa(W, Param) && isa(b, Param)
# end

test_anon(identity)

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

end
