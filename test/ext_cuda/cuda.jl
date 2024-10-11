using Flux, Test, Zygote
using Flux: cpu, gpu
using Statistics: mean
using LinearAlgebra: I, cholesky, Cholesky, Adjoint
using SparseArrays: sparse, SparseMatrixCSC, AbstractSparseArray
using CUDA
CUDA.allowscalar(false)

@testset "CUDA" begin
  x = randn(5, 5)
  cx = gpu(x)
  @test cx isa CuArray

  @test Flux.onecold(gpu([1.0, 2.0, 3.0])) == 3

  x = Flux.onehotbatch([1, 2, 3], 1:3)
  cx = gpu(x)
  @test cx isa Flux.OneHotMatrix && cx.indices isa CuArray
  @test (cx .+ 1) isa CuArray

  m = Chain(Dense(10, 5, tanh), Dense(5, 2), softmax)
  cm = gpu(m)

  @test all(p isa CuArray for p in Flux.trainables(cm))
  @test cm(gpu(rand(10, 10))) isa CuArray{Float32,2}

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

@testset "onehot gpu" begin
  y = Flux.onehotbatch(ones(3), 1:2) |> gpu;
  @test (repr("text/plain", y); true)

  gA = rand(3, 2) |> gpu;
  @test gradient(A -> sum(A * y), gA)[1] isa CuArray

  # construct from CuArray
  x = [1, 3, 2]
  y = Flux.onehotbatch(x, 0:3)
  
  # https://github.com/FluxML/OneHotArrays.jl/issues/16
  y2 = Flux.onehotbatch(x |> gpu, 0:3)
  @test y2.indices isa CuArray
  @test y2 |> cpu == y
end

@testset "onecold gpu" begin
  y = Flux.onehotbatch(ones(3), 1:10) |> gpu;
  l = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
  @test Flux.onecold(y) isa CuArray
  @test y[3,:] isa CuArray
  @test Flux.onecold(y, l) == ['a', 'a', 'a']
end

@testset "onehot forward map to broadcast" begin
  oa = Flux.OneHotArray(rand(1:10, 5, 5), 10) |> gpu
  @test all(map(identity, oa) .== oa)
  @test all(map(x -> 2 * x, oa) .== 2 .* oa)
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

  @testset "isbits array types" begin
    struct SimpleBits
      field::Int32
    end

    @test gpu((;a=ones(1))).a isa CuVector{Float32}
    @test gpu((;a=['a', 'b', 'c'])).a isa CuVector{Char}
    @test gpu((;a=[SimpleBits(1)])).a isa CuVector{SimpleBits}
  end
end

@testset "gpu(cpu(x)) inside gradient" begin
  a = randn(Float32, 4, 4)
  ca = cu(a)

  # Trivial functions
  @test gradient(x -> sum(abs, gpu(x)), a)[1] isa Matrix
  @test gradient(x -> sum(gpu(x)), a)[1] isa Matrix
  @test_broken gradient(x -> sum(gpu(x)), a')[1] isa Matrix  # sum(::Adjoint{T,CuArray}) makes a Fill
  @test gradient(x -> sum(abs, cpu(x)), ca)[1] isa CuArray
  # This test should really not go through indirections and pull out Fills for efficiency
  # but we forcefully materialise. TODO: remove materialising CuArray here
  @test gradient(x -> sum(cpu(x)), ca)[1] isa CuArray # This involves FillArray, which should be GPU compatible
  @test gradient(x -> sum(cpu(x)), ca')[1] isa CuArray

  # Even more trivial: no movement
  @test gradient(x -> sum(abs, cpu(x)), a)[1] isa Matrix
  @test gradient(x -> sum(abs, cpu(x)), a')[1] isa Matrix
  @test gradient(x -> sum(cpu(x)), a)[1] isa typeof(gradient(sum, a)[1]) # FillArray
  @test gradient(x -> sum(abs, gpu(x)), ca)[1] isa CuArray
  @test gradient(x -> sum(abs, gpu(x)), ca')[1] isa CuArray

  # More complicated, Array * CuArray is an error
  g0 = gradient(x -> sum(abs, (a * (a * x))), a)[1]
  @test g0 ≈ gradient(x -> sum(abs, cpu(ca * gpu(a * x))), a)[1]
  @test cu(g0) ≈ gradient(x -> sum(abs, gpu(a * cpu(ca * x))), ca)[1]
  @test gradient(x -> sum(gpu(cpu(x))), a)[1] isa Matrix
  @test gradient(x -> sum(gpu(cpu(x))), ca)[1] isa CuArray

  g4 = gradient(x -> sum(a * (a' * x)), a)[1]  # no abs, one adjoint
  @test g4 ≈ gradient(x -> sum(cpu(ca * gpu(a' * x))), a)[1]
  @test cu(g4) ≈ gradient(x -> sum(gpu(a * cpu(ca' * x))), ca)[1]

  # Scalar indexing of an array, needs OneElement to transfer to GPU
  # https://github.com/FluxML/Zygote.jl/issues/1005
  @test_broken gradient(x -> cpu(2 .* gpu(x))[1], Float32[1,2,3]) == ([2,0,0],)
  @test_broken gradient(x -> cpu(gpu(x) * gpu(x))[1,2], Float32[1 2 3; 4 5 6; 7 8 9]) == ([2 6 8; 0 2 0; 0 3 0],)
end

@testset "gpu(x) and cpu(x) on structured arrays" begin
  @test cpu(1:3) isa UnitRange
  @test cpu(range(1, 3, length = 4)) isa AbstractRange

  # OneElement isn't GPU compatible
  g1 = Zygote.OneElement(1, (2,3), axes(ones(4,5)))
  @test cpu(g1) isa Zygote.OneElement

  g2 = Zygote.Fill(1f0, 2)
  @test cpu(g2) isa Zygote.FillArrays.AbstractFill

  g3 = transpose(Float32[1 2; 3 4])
  @test parent(cpu(g3)) isa Matrix{Float32}

  @testset "Sparse Arrays" begin
    @test cpu(sparse(rand(3,3))) isa SparseMatrixCSC
    a = sparse(rand(3,3))
    @test cpu(a) === a
    @test gpu(sparse(rand(3,3))) isa CUDA.CUSPARSE.CuSparseMatrixCSC
  end

  # Check that gpu() converts these to CuArrays. This a side-effect of using the same functions
  # in gpu() as in the gradient of cpu(). A different design could avoid having gpu() used alone
  # move these, if that turns out to be desirable.
  @test gpu(g1) isa CuArray
  @test gpu(g1) ≈ cu(Matrix(g1))
  @test gpu(g2) isa CuArray
  @test gpu(g2) ≈ cu(Vector(g2))
  @test parent(gpu(g3)) isa CuArray


  #Issue #2116
  struct A2116
    x::Int
    y::Int
  end
  x = [A2116(1,1), A2116(2,2)]
  xgpu = gpu(x)
  @test xgpu isa CuVector{A2116}
  @test cpu(xgpu) isa Vector{A2116}
  @test cpu(gpu([CartesianIndex(1)])) isa Vector{CartesianIndex{1}}
end

@testset "gpu(::DataLoader)" begin
  X = randn(Float64, 3, 33)
  pre1 = Flux.DataLoader(X |> gpu; batchsize=13, shuffle=false)
  post1 = Flux.DataLoader(X; batchsize=13, shuffle=false) |> gpu
  rev1 = pre1 |> cpu  # inverse operation
  for epoch in 1:2
    for (p, q, a) in zip(pre1, post1, rev1)
      @test p isa CuArray{Float32}
      @test q isa CuArray{Float32}
      @test p ≈ q
      @test a isa Array{Float32}
      @test a ≈ Array(p)
    end
  end

  Y = Flux.onehotbatch(rand(0:2, 33), 0:2)
  pre2 = Flux.DataLoader((x=X, y=Y) |> gpu; batchsize=7, shuffle=false)
  post2 = Flux.DataLoader((x=X, y=Y); batchsize=7, shuffle=false) |> gpu
  for (p, q) in zip(pre2, post2)
    @test p.x == q.x
    @test_broken p.y == q.y  # https://github.com/FluxML/OneHotArrays.jl/issues/28 -- MethodError: getindex(::OneHotArrays.OneHotMatrix{UInt32, CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}}, ::Int64, ::Int64) is ambiguous
  end

  @test collect(pre2) isa Vector{<:NamedTuple{(:x, :y)}}
  @test collect(post2) isa Vector{<:NamedTuple{(:x, :y)}}  # collect makes no sense, but check eltype?

  # @test_throws Exception gpu(((x = Flux.DataLoader(X), y = Y),))
end