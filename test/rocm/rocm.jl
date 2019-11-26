using Flux, Test
using Flux.ROCArrays
using Flux: rocgpu

@info "Testing ROCm GPU Support"

@testset "ROCArrays" begin

ROCArrays.allowscalar(false)

x = randn(5, 5)
cx = rocgpu(x)
@test cx isa ROCArray

@test Flux.onecold(rocgpu([1.0, 2.0, 3.0])) == 3

x = Flux.onehotbatch([1, 2, 3], 1:3)
cx = rocgpu(x)
@test cx isa Flux.OneHotMatrix && cx.data isa ROCArray
@test (cx .+ 1) isa ROCArray

m = Chain(Dense(10, 5, tanh), Dense(5, 2), softmax)
cm = rocgpu(m)

@test all(p isa ROCArray for p in params(cm))
@test cm(rocgpu(rand(10, 10))) isa ROCArray{Float32,2}

x = [1,2,3]
cx = rocgpu(x)
@test Flux.crossentropy(x,x) ≈ Flux.crossentropy(cx,cx)
@test Flux.crossentropy(x,x, weight=1.0) ≈ Flux.crossentropy(cx,cx, weight=1.0)
@test Flux.crossentropy(x,x, weight=[1.0;2.0;3.0]) ≈ Flux.crossentropy(cx,cx, weight=cu([1.0;2.0;3.0]))

x = σ.([-1.1491, 0.8619, 0.3127])
y = [1, 1, 0.]
@test Flux.binarycrossentropy.(x,y) ≈ Flux.binarycrossentropy.(cu(x),cu(y))

xs = rand(5, 5)
ys = Flux.onehotbatch(1:5,1:5)
@test collect(cu(xs) .+ cu(ys)) ≈ collect(xs .+ ys)

c = rocgpu(Conv((2,2),3=>4))
x = rocgpu(rand(10, 10, 3, 2))
l = c(rocgpu(rand(10,10,3,2)))
@test gradient(x -> sum(c(x)), x)[1] isa ROCArray

c = rocgpu(CrossCor((2,2),3=>4))
x = rocgpu(rand(10, 10, 3, 2))
l = c(rocgpu(rand(10,10,3,2)))
@test gradient(x -> sum(c(x)), x)[1] isa ROCArray

end

@testset "onecold rocgpu" begin
  y = Flux.onehotbatch(ones(3), 1:10) |> rocgpu;
  @test Flux.onecold(y) isa ROCArray
  @test y[3,:] isa ROCArray
end

if isdefined(ROCArrays, :MIOpen)
  @info "Testing Flux/MIOpen"
  include("miopen.jl")
else
  @warn "MIOpen unavailable, not testing GPU DNN support"
end
