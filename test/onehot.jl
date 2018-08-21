using Flux:argmax
using Test

@testset "argmax" begin
  a = [1, 2, 5, 3.]
  A = [1 20 5; 2 7 6; 3 9 10; 2 1 14]
  labels = ['A', 'B', 'C', 'D']

  @test argmax(a) == 3
  @test argmax(A) == CartesianIndex(1, 2)
  @test argmax(a, labels) == 'C'
  @test argmax(A, labels) == ['C', 'A', 'D']
end
