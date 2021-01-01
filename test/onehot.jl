using Flux:onecold
using Test

@testset "onecold" begin
  a = [1, 2, 5, 3.]
  A = [1 20 5; 2 7 6; 3 9 10; 2 1 14]
  labels = ['A', 'B', 'C', 'D']

  @test onecold(a) == 3
  @test onecold(A) == [3, 1, 4]
  @test onecold(a, labels) == 'C'
  @test onecold(A, labels) == ['C', 'A', 'D']

  data = [:b, :a, :c]
  labels = [:a, :b, :c]
  hot = Flux.onehotbatch(data, labels)
  cold = onecold(hot, labels)

  @test cold == data
end

@testset "onehotbatch indexing" begin
  y = Flux.onehotbatch(ones(3), 1:10)
  @test y[:,1] isa Flux.OneHotVector
  @test y[:,:] isa Flux.OneHotMatrix
end

@testset "abstractmatrix onehotvector multiplication" begin
  A = [1 3 5; 2 4 6; 3 6 9]
  b1 = Flux.OneHotVector{eltype(A), 3}(1)
  b2 = Flux.OneHotVector{eltype(A), 5}(3)

  @test A*b1 == A[:,1]
  @test_throws DimensionMismatch A*b2
end