using Flux, CuArrays, Test

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
