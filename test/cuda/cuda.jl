using Flux, Test
using Flux.CUDA
using Flux: cpu, gpu
using Statistics: mean
using LinearAlgebra: I, cholesky, Cholesky

@info "Testing GPU Support"

@testset "CUDA" begin

CUDA.allowscalar(false)

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

x = [1.,2.,3.]
cx = gpu(x)
@test Flux.crossentropy(x,x) ≈ Flux.crossentropy(cx,cx) 
@test Flux.crossentropy(x,x, agg=identity) ≈ Flux.crossentropy(cx,cx, agg=identity) |> cpu
@test Flux.crossentropy(x,x, agg=x->mean([1.0;2.0;3.0].*x)) ≈ Flux.crossentropy(cx,cx, agg=x->mean(cu([1.0;2.0;3.0]).*x))

x = [-1.1491, 0.8619, 0.3127]
y = [1, 1, 0.]
@test Flux.bce_loss(σ.(x), y) ≈ Flux.bce_loss(cu(σ.(x)), cu(y))
@test Flux.logitbce_loss(x, y) ≈ Flux.logitbce_loss(cu(x), cu(y))

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

@testset "restructure gpu" begin
  dudt = Dense(1,1) |> gpu
  p,re = Flux.destructure(dudt)
  foo(x) = sum(re(p)(x))
  @test gradient(foo, cu(rand(1)))[1] isa CuArray
end

@testset "GPU functors" begin
  @testset "Cholesky" begin
    M = 2.0*I(10) |> collect
    Q = cholesky(M)
    Q_gpu = Q |> gpu
    @test Q_gpu isa Cholesky{<:Any,<:CuArray}
    Q_cpu = Q_gpu |> cpu
    @test Q_cpu == cholesky(eltype(Q_gpu).(M))
  end
end

if CUDA.has_cudnn()
  @info "Testing Flux/CUDNN"
  include("cudnn.jl")
  include("curnn.jl")
  include("layers.jl")
else
  @warn "CUDNN unavailable, not testing GPU DNN support"
end
