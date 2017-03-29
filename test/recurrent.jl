using Flux: stateless

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
  xs = [rand(10) for _ = 1:3]
  _, ys = apply(stateless(unroll1(r)), xs, (squeeze(r.y.x, 1),))
  ru = unroll(r, 3)
  @test ru(Seq(xs)) == ys
end
