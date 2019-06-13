using Flux, CuArrays, Test

@testset "RNN" begin
  @testset for R in [RNN, GRU, LSTM]
    rnn = R(10, 5)
    curnn = mapleaves(gpu, rnn)
    @testset for batch_size in (1, 5)
      Flux.reset!(rnn)
      Flux.reset!(curnn)
      x = batch_size == 1 ?
        param(rand(10)) :
        param(rand(10,batch_size))
      cux = gpu(x)
      y = (rnn(x); rnn(x))
      cuy = (curnn(cux); curnn(cux))

      @test y ≈ collect(cuy)
      @test haskey(Flux.CUDA.descs, curnn.cell)

      #Δ = randn(size(y))

      #Flux.back!(y, Δ)
      #Flux.back!(cuy, gpu(Δ))

      @test x ≈ collect(cux)
      @test rnn.cell.Wi ≈ collect(curnn.cell.Wi)
      @test rnn.cell.Wh ≈ collect(curnn.cell.Wh)
      @test rnn.cell.b ≈ collect(curnn.cell.b)
      @test rnn.cell.h ≈ collect(curnn.cell.h)
      if isdefined(rnn.cell, :c)
        @test rnn.cell.c ≈ collect(curnn.cell.c)
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
end
