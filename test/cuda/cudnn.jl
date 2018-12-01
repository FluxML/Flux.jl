<<<<<<< HEAD
using Flux, CuArrays, Test

@info "Testing Flux/CUDNN"

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
      @test rnn.cell.b.grad ≈ coltracklect(curnn.cell.b.grad)
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
=======
using Flux, Flux.Tracker, CuArrays, Test
using Flux.Tracker: TrackedArray, data

@testset "CUDNN BatchNorm" begin
    @testset "4D Input" begin
        x = TrackedArray(Float64.(collect(reshape(1:12, 2, 2, 3, 1))))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y = m(x)
        cy = cm(cx)

        @test cy isa TrackedArray{Float32,4,CuArray{Float32,4}}

        @test cpu(data(cy)) ≈ data(y)

        g = rand(size(y)...)
        Flux.back!(y, g)
        Flux.back!(cy, gpu(g))

        @test m.γ.grad ≈ cpu(cm.γ.grad)
        @test m.β.grad ≈ cpu(cm.β.grad)
        @test x.grad ≈ cpu(x.grad)
    end

    @testset "2D Input" begin
        x = TrackedArray(Float64.(collect(reshape(1:12, 3, 4))))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y = m(x)
        cy = cm(cx)

        @test cy isa TrackedArray{Float32,2,CuArray{Float32,2}}

        @test cpu(data(cy)) ≈ data(y)

        g = rand(size(y)...)
        Flux.back!(y, g)
        Flux.back!(cy, gpu(g))

        @test m.γ.grad ≈ cpu(cm.γ.grad)
        @test m.β.grad ≈ cpu(cm.β.grad)
        @test x.grad ≈ cpu(x.grad)
>>>>>>> a32c8a2e60870d49475719a36c74afed305e370a
    end
end

@testset "CNN" begin
  cnn = Conv((3, 3), 1=>10, pad = 1)
  cucnn = cnn |> gpu
  x = rand(10, 10, 1, 1)
  cux = x |> gpu
  y = cnn(x)
  cuy = cucnn(cux)
  Δ = rand(size(y))

  @test y.data ≈ collect(cuy.data)

  Flux.back!(y, Δ)
  Flux.back!(cuy, gpu(Δ))

  @test cnn.weight.data ≈ collect(cucnn.weight.data)
  @test cnn.bias.data ≈ collect(cucnn.bias.data)
end
