using Flux, Test
using Flux.CuArrays
using Flux: gpu

@info "Testing GPU Support"

@testset "CuArrays" begin

CuArrays.allowscalar(false)

x = randn(5, 5)
cx = gpu(x)
@test cx isa CuArray

@test Flux.onecold(gpu([1.0, 2.0, 3.0])) == 3

x = Flux.onehotbatch([1, 2, 3], 1:3)
cx = gpu(x)
@test cx isa Flux.OneHotMatrix && cx.data isa CuArray
@test (cx .+ 1) isa CuArray

m = Chain(Dense(10, 5, tanh), Dense(5, 2), softmax)
cm = gpu(m)

@test all(p isa CuArray for p in params(cm))
@test cm(gpu(rand(10, 10))) isa CuArray{Float32,2}

x = rand(3) 
y = rand(3) 
cx = gpu(x)
cy = gpu(y)
@test Flux.mse(cx, cy) ≈ Flux.mse(x, y)
x = [1,2,3]
cx = gpu(x)
@test Flux.crossentropy(x,x) ≈ Flux.crossentropy(cx,cx)
@test Flux.crossentropy(x,x, weight=1.0) ≈ Flux.crossentropy(cx,cx, weight=1.0)
@test Flux.crossentropy(x,x, weight=[1.0;2.0;3.0]) ≈ Flux.crossentropy(cx,cx, weight=cu([1.0;2.0;3.0]))
x = [-1.1491, 0.8619, 0.3127]
y = [1, 1, 0]
cx = gpu(x)
scx = σ.(cx)
cy = gpu(y)
@test_broken Flux.binarycrossentropy.(σ.(x), y) ≈ Flux.binarycrossentropy.(scx, cy)
@test_broken Flux.logitbinarycrossentropy.(x,y) ≈ Flux.logitbinarycrossentropy.(cx, cy)

xs = rand(5, 5)
# fails due to scalar getindex
@test_broken collect(Flux.normalise(cu(xs))) ≈ Flux.normalise(xs)

xs = rand(5, 5)
ys = Flux.onehotbatch(1:5,1:5)
@test collect(cu(xs) .+ cu(ys)) ≈ collect(xs .+ ys)

c = gpu(Conv((2,2),3=>4))
x = gpu(rand(10, 10, 3, 2))
l = c(gpu(rand(10,10,3,2)))
@test gradient(x -> sum(c(x)), x)[1] isa CuArray

c = gpu(CrossCor((2,2),3=>4))
x = gpu(rand(10, 10, 3, 2))
l = c(gpu(rand(10,10,3,2)))
@test gradient(x -> sum(c(x)), x)[1] isa CuArray

end

@testset "onecold gpu" begin
  y = Flux.onehotbatch(ones(3), 1:10) |> gpu;
  @test Flux.onecold(y) isa CuArray
  @test y[3,:] isa CuArray
end

if CuArrays.libcudnn != nothing
  @info "Testing Flux/CUDNN"
  include("cudnn.jl")
  include("curnn.jl")
end
