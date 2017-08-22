using DataFlow, MacroTools
using Flux: squeeze, unsqueeze, stack
using Flux.Compiler: @net, graph
using DataFlow: Line, Frame

@net type Affine
  W
  b
  x -> x*W .+ b
end

Affine(in::Integer, out::Integer; init = Flux.initn) =
  Affine(init(in, out), init(1, out))

@net type TLP
  first
  second
  function (x)
    l1 = σ.(first(x))
    l2 = softmax(second(l1))
  end
end

@net type Recurrent
  Wxy; Wyy; by
  y
  function (x)
    y = tanh.( x * Wxy .+ y{-1} * Wyy .+ by )
  end
end

Recurrent(in, out; init = Flux.initn) =
  Recurrent(init((in, out)), init((out, out)), init(1, out), init(1, out))

syntax(v::Vertex) = prettify(DataFlow.syntax(v))
syntax(x) = syntax(graph(x))

@testset "Compiler" begin

xs = randn(1, 10)
d = Affine(10, 20)

@test d(xs) ≈ (xs*d.W + d.b)

d1 = @net x -> x * d.W + d.b

let
  @capture(syntax(d), _Frame(_Line((+).(x_[1] * W_, b_))))
  @test isa(x, DataFlow.Input) && W isa Array && b isa Array
end

let a1 = Affine(10, 20), a2 = Affine(20, 15)
  tlp = TLP(a1, a2)
  @test tlp(xs) ≈ softmax(a2(σ.(a1(xs))))
  @test Flux.Compiler.interpmodel(tlp, xs) ≈ softmax(a2(σ.(a1(xs))))
end

let tlp = TLP(Affine(10, 21), Affine(20, 15))
  e = try
    Flux.Compiler.interpmodel(tlp, rand(1, 10))
  catch e
    e
  end
  @test e.trace[end].func == :TLP
  @test e.trace[end-1].func == Symbol("Affine")
end

function apply(model, xs, state)
  ys = similar(xs, 0)
  for x in xs
    state, y = model(state, x)
    push!(ys, y)
  end
  state, ys
end

@testset "RNN unrolling" begin
  r = Recurrent(10, 5)
  xs = [rand(1, 10) for _ = 1:3]
  _, ys = apply(Flux.Compiler.unroll1(r).model, xs, (r.y,))
  @test ys[1] == tanh.(xs[1] * r.Wxy .+ r.y * r.Wyy .+ r.by)
  ru = Flux.Compiler.unroll(r, 3)
  ru(unsqueeze(stack(squeeze.(xs))))[1] == squeeze.(ys)
end

end
