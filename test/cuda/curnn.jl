using Flux, CUDA, Test

Wxh = randn(5, 2)
Whh = randn(5, 5)
b   = randn(5)

function rnn2((h,), x)
  h = tanh.(Wxh * x .+ Whh * h .+ b)
  return (h,), h
end

x = rand(2) # dummy data
h = (rand(5),)  # initial hidden state

h, y = rnn2(h, x)

rnn3 = Flux.RNNCell(2, 5)

x = rand(Float32, 2) # dummy data
h = (rand(Float32, 5),)  # initial hidden state
h, y = rnn3(h, x)
@code_warntype rnn3(h, x)

rnn4 = RNN(2,5)
h, y = rnn4(x)
@code_warntype rnn4(x)



@testset for R in [RNN, GRU, LSTM]
  m = R(10, 5) |> gpu
  x = gpu(rand(10))
  (m̄,) = gradient(m -> sum(m(x)), m)
  Flux.reset!(m)
  θ = gradient(() -> sum(m(x)), params(m))
  @test x isa CuArray
  @test θ[m.cell.Wi] isa CuArray
  @test collect(m̄[].cell.Wi) == collect(θ[m.cell.Wi])
end

@testset "RNN" begin
  @testset for R in [RNN, GRU, LSTM], batch_size in (1, 5)
    rnn = R(10, 5)
    curnn = fmap(gpu, rnn)

    Flux.reset!(rnn)
    Flux.reset!(curnn)
    x = batch_size == 1 ?
      rand(Float32, 10) :
      rand(Float32, 10, batch_size)
    cux = gpu(x)

    y, back = pullback((r, x) -> r(x), rnn, x)
    cuy, cuback = pullback((r, x) -> r(x), curnn, cux)

    @test y ≈ collect(cuy)

    ȳ = randn(size(y))
    m̄, x̄ = back(ȳ)
    cum̄, cux̄ = cuback(gpu(ȳ))

    @test x̄ ≈ collect(cux̄)
    @test m̄[].cell.Wi ≈ collect(cum̄[].cell.Wi)
    @test m̄[].cell.Wh ≈ collect(cum̄[].cell.Wh)
    @test m̄[].cell.b ≈ collect(cum̄[].cell.b)
    if m̄[].state isa Tuple
      for (x, cx) in zip(m̄[].state, cum̄[].state)
        @test x ≈ collect(cx)
      end
    else
      @test m̄[].state ≈ collect(cum̄[].state)
    end

    Flux.reset!(rnn)
    Flux.reset!(curnn)
    ohx = batch_size == 1 ?
      Flux.onehot(rand(1:10), 1:10) :
      Flux.onehotbatch(rand(1:10, batch_size), 1:10)
    cuohx = gpu(ohx)
    y = (rnn(ohx); rnn(ohx))

    cuy = (curnn(cuohx); curnn(cuohx))
    @test y ≈ collect(cuy)  
  end
end
