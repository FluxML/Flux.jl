using Flux, CuArrays, Base.Test

info("Testing Flux/CUDNN")

@testset "RNN" begin
  @testset for R in [RNN, GRU, LSTM]
    x = param(rand(10,5))
    cux = cu(x)
    rnn = R(10, 5)
    curnn = mapleaves(cu, rnn)
    y = (rnn(x); rnn(x))
    cuy = (curnn(cux); curnn(cux))

    @test y.data ≈ collect(cuy.data)
    @test haskey(Flux.CUDA.descs, curnn.cell)

    Δ = randn(size(y))

    Flux.back!(y, Δ)
    Flux.back!(cuy, cu(Δ))

    @test x.grad ≈ collect(cux.grad)
    @test rnn.cell.Wi.grad ≈ collect(curnn.cell.Wi.grad)
    @test rnn.cell.Wh.grad ≈ collect(curnn.cell.Wh.grad)
    @test rnn.cell.b.grad ≈ collect(curnn.cell.b.grad)
    @test rnn.cell.h.grad ≈ collect(curnn.cell.h.grad)
    if isdefined(rnn.cell, :c)
      @test rnn.cell.c.grad ≈ collect(curnn.cell.c.grad)
    end
  end
end
