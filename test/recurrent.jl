using Flux: Recurrent

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
  _, ys = apply(Flux.Compiler.unroll1(r).model, xs, (r.y.x,))
  @test ys[1] == tanh(xs[1] * r.Wxy.x .+ r.y.x * r.Wyy.x .+ r.by.x)
  ru = Flux.Compiler.unroll(r, 3)
  ru(unsqueeze(stack(squeeze.(xs))))[1] == squeeze.(ys)
end
