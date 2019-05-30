using Flux, Flux.Tracker, CuArrays, Test
using Flux: gpu

@info "Testing GPU Support"

@testset "CuArrays" begin

CuArrays.allowscalar(false)

x = param(randn(5, 5))
cx = gpu(x)
@test cx isa TrackedArray && cx.data isa CuArray

@test Flux.onecold(param(gpu([1.,2.,3.]))) == 3

x = Flux.onehotbatch([1, 2, 3], 1:3)
cx = gpu(x)
@test cx isa Flux.OneHotMatrix && cx.data isa CuArray
@test (cx .+ 1) isa CuArray

m = Chain(Dense(10, 5, tanh), Dense(5, 2), softmax)
cm = gpu(m)

@test all(p isa TrackedArray && p.data isa CuArray for p in params(cm))
@test cm(gpu(rand(10, 10))) isa TrackedArray{Float32,2,CuArray{Float32,2}}

x = [1,2,3]
cx = gpu(x)
@test Flux.crossentropy(x,x) ≈ Flux.crossentropy(cx,cx)

xs = param(rand(5,5))
ys = Flux.onehotbatch(1:5,1:5)
@test collect(cu(xs) .+ cu(ys)) ≈ collect(xs .+ ys)

c = gpu(Conv((2,2),3=>4))
l = c(gpu(rand(10,10,3,2)))
Flux.back!(sum(l))

c = gpu(CrossCor((2,2),3=>4))
l = c(gpu(rand(10,10,3,2)))
Flux.back!(sum(l))

end

@testset "onecold gpu" begin
  y = Flux.onehotbatch(ones(3), 1:10) |> gpu;
  @test Flux.onecold(y) isa CuArray
  @test y[3,:] isa CuArray
end

@testset "Jacobian on GPU" begin
  # https://github.com/FluxML/Tracker.jl/pull/33
  @test collect(jacobian(identity, gpu([0.0, 0.0]))) == [1 0; 0 1]
  @test collect(gradient(x -> sum(jacobian(y -> y .^ 2, x) .^ 2),
                         gpu([1.0, 2.0, 3.0]))) == [8.0, 16.0, 24.0]
end

if CuArrays.libcudnn != nothing
    @info "Testing Flux/CUDNN"
    include("cudnn.jl")
    if !haskey(ENV, "CI_DISABLE_CURNN_TEST")
      include("curnn.jl")
    end
end
