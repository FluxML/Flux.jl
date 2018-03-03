using Flux, CuArrays, Base.Test

info("Testing Flux/CUDNN")

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

      @test y.data ≈ collect(cuy.data)
      @test haskey(Flux.CUDA.descs, curnn.cell)

      Δ = randn(size(y))

      Flux.back!(y, Δ)
      Flux.back!(cuy, gpu(Δ))

      @test x.grad ≈ collect(cux.grad)
      @test rnn.cell.Wi.grad ≈ collect(curnn.cell.Wi.grad)
      @test rnn.cell.Wh.grad ≈ collect(curnn.cell.Wh.grad)
      @test rnn.cell.b.grad ≈ collect(curnn.cell.b.grad)
      @test rnn.cell.h.grad ≈ collect(curnn.cell.h.grad)
      if isdefined(rnn.cell, :c)
        @test rnn.cell.c.grad ≈ collect(curnn.cell.c.grad)
      end

      Flux.reset!(rnn)
      Flux.reset!(curnn)
      ohx = batch_size == 1 ?
        Flux.onehot(rand(1:10), 1:10) :
        Flux.onehotbatch(rand(1:10, batch_size), 1:10)
      cuohx = gpu(ohx)
      y = (rnn(ohx); rnn(ohx))
      cuy = (curnn(cuohx); curnn(cuohx))

      @test y.data ≈ collect(cuy.data)
    end
  end
end
