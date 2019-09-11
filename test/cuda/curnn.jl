using Flux, CuArrays, Test
using Flux: forward

@testset "RNN" begin
  @testset for R in [RNN, GRU, LSTM], batch_size in (1, 5)
    rnn = R(10, 5)
    curnn = mapleaves(gpu, rnn)

    Flux.reset!(rnn)
    Flux.reset!(curnn)
    x = batch_size == 1 ?
      rand(10) :
      rand(10, batch_size)
    cux = gpu(x)

    y, back = forward((r, x) -> (r(x)), rnn, x)
    cuy, cuback = forward((r, x) -> (r(x)), curnn, cux)

    @test y ≈ collect(cuy)
    @test haskey(Flux.CUDA.descs, curnn.cell)

    ȳ = randn(size(y))
    m̄, x̄ = back(ȳ)
    cum̄, cux̄ = cuback(gpu(ȳ))

    m̄[].cell[].Wi

    m̄[].state
    cum̄[].state

    @test x̄ ≈ collect(cux̄)
    @test m̄[].cell[].Wi ≈ collect(cum̄[].cell[].Wi)
    @test m̄[].cell[].Wh ≈ collect(cum̄[].cell[].Wh)
    @test m̄[].cell[].b ≈ collect(cum̄[].cell[].b)
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
