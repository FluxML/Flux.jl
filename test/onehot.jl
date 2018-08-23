using Flux:onecold
using Test

@testset "argmax" begin
  a = [1, 2, 5, 3.]
  A = [1 20 5; 2 7 6; 3 9 10; 2 1 14]
  labels = ['A', 'B', 'C', 'D']
  
  @test onecold(a) == 3
  @test onecold(A) == [3, 1, 4]
  @test onecold(a, labels) == 'C'
  @test onecold(A, labels) == ['C', 'A', 'D']
end
