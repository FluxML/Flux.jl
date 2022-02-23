using Flux: onehot, onehotbatch, onecold
using Test

@testset "onehot constructors" begin
  @test onehot(20, 10:10:30) == [false, true, false]
  @test onehot(20, (10,20,30)) == [false, true, false]
  @test onehot(40, (10,20,30), 20) == [false, true, false]

  @test_throws Exception onehot('d', 'a':'c')
  @test_throws Exception onehot(:d, (:a, :b, :c))
  @test_throws Exception onehot('d', 'a':'c', 'e')
  @test_throws Exception onehot(:d, (:a, :b, :c), :e)

  @test onehotbatch([20, 10], 10:10:30) == Bool[0 1; 1 0; 0 0]
  @test onehotbatch([20, 10], (10,20,30)) == Bool[0 1; 1 0; 0 0]
  @test onehotbatch([40, 10], (10,20,30), 20) == Bool[0 1; 1 0; 0 0]

  @test onehotbatch("abc", 'a':'c') == Bool[1 0 0; 0 1 0; 0 0 1]
  @test onehotbatch("zbc", ('a', 'b', 'c'), 'a') == Bool[1 0 0; 0 1 0; 0 0 1]

  @test onehotbatch([10, 20], [30, 40, 50], 30) == Bool[1 1; 0 0; 0 0]

  @test_throws Exception onehotbatch([:a, :d], [:a, :b, :c])
  @test_throws Exception onehotbatch([:a, :d], (:a, :b, :c))
  @test_throws Exception onehotbatch([:a, :d], [:a, :b, :c], :e)
  @test_throws Exception onehotbatch([:a, :d], (:a, :b, :c), :e)
  @test_throws Exception onehotbatch([:a, :e], (:a, :b, :c), :d)

  floats = (0.0, -0.0, NaN, -NaN, Inf, -Inf)
  @test onecold(onehot(0.0, floats)) == 1
  @test onecold(onehot(-0.0, floats)) == 2  # as it uses isequal
  @test onecold(onehot(Inf, floats)) == 5
end

@testset "onecold" begin
  a = [1, 2, 5, 3.]
  A = [1 20 5; 2 7 6; 3 9 10; 2 1 14]
  labels = ['A', 'B', 'C', 'D']

  @test onecold(a) == 3
  @test onecold(A) == [3, 1, 4]
  @test onecold(a, labels) == 'C'
  @test onecold(a, Tuple(labels)) == 'C'
  @test onecold(A, labels) == ['C', 'A', 'D']
  @test onecold(A, Tuple(labels)) == ['C', 'A', 'D']

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
  v = [1, 2, 3, 4, 5]
  X = reshape(v, (5, 1))
  b1 = Flux.OneHotVector(1, 3)
  b2 = Flux.OneHotVector(3, 5)

  @test A * b1 == A[:,1]
  @test b1' * A == Array(b1') * A
  @test A' * b1 == A' * Array(b1)
  @test v' * b2 == v' * Array(b2)
  @test transpose(X) * b2 == transpose(X) * Array(b2)
  @test transpose(v) * b2 == transpose(v) * Array(b2)
  @test_throws DimensionMismatch A*b2
end

@testset "AbstractMatrix-OneHotMatrix multiplication" begin
  A = [1 3 5; 2 4 6; 3 6 9]
  v = [1, 2, 3, 4, 5]
  X = reshape(v, (5, 1))
  b1 = Flux.OneHotMatrix([1, 1, 2, 2], 3)
  b2 = Flux.OneHotMatrix([2, 4, 1, 3], 5)
  b3 = Flux.OneHotMatrix([1, 1, 2], 4)
  b4 = reshape(Flux.OneHotMatrix([1 2 3; 2 2 1], 3), 3, :)
  b5 = reshape(b4, 6, :)
  b6 = reshape(Flux.OneHotMatrix([1 2 2; 2 2 1], 2), 3, :)
  b7 = reshape(Flux.OneHotMatrix([1 2 3; 1 2 3], 3), 6, :)

  @test A * b1 == A[:,[1, 1, 2, 2]]
  @test b1' * A == Array(b1') * A
  @test A' * b1 == A' * Array(b1)
  @test A * b3' == A * Array(b3')
  @test transpose(X) * b2 == transpose(X) * Array(b2)
  @test A * b4 == A[:,[1, 2, 2, 2, 3, 1]]
  @test A * b5' == hcat(A[:,[1, 2, 3, 3]], A[:,1]+A[:,2], zeros(Int64, 3))
  @test A * b6 == hcat(A[:,1], 2*A[:,2], A[:,2], A[:,1]+A[:,2])
  @test A * b7' == A[:,[1, 2, 3, 1, 2, 3]]

  @test_throws DimensionMismatch A*b1'
  @test_throws DimensionMismatch A*b2
  @test_throws DimensionMismatch A*b2'
  @test_throws DimensionMismatch A*b6'
  @test_throws DimensionMismatch A*b7
end

@testset "OneHotArray" begin
  using Flux: OneHotArray, OneHotVector, OneHotMatrix, OneHotLike
  
  ov = OneHotVector(rand(1:10), 10)
  ov2 = OneHotVector(rand(1:11), 11)
  om = OneHotMatrix(rand(1:10, 5), 10)
  om2 = OneHotMatrix(rand(1:11, 5), 11)
  oa = OneHotArray(rand(1:10, 5, 5), 10)

  # sizes
  @testset "Base.size" begin
    @test size(ov) == (10,)
    @test size(om) == (10, 5)
    @test size(oa) == (10, 5, 5)
  end

  @testset "Indexing" begin
    # vector indexing
    @test ov[3] == (ov.indices == 3)
    @test ov[:] == ov
    
    # matrix indexing
    @test om[3, 3] == (om.indices[3] == 3)
    @test om[:, 3] == OneHotVector(om.indices[3], 10)
    @test om[3, :] == (om.indices .== 3)
    @test om[:, :] == om

    # array indexing
    @test oa[3, 3, 3] == (oa.indices[3, 3] == 3)
    @test oa[:, 3, 3] == OneHotVector(oa.indices[3, 3], 10)
    @test oa[3, :, 3] == (oa.indices[:, 3] .== 3)
    @test oa[3, :, :] == (oa.indices .== 3)
    @test oa[:, 3, :] == OneHotMatrix(oa.indices[3, :], 10)
    @test oa[:, :, :] == oa

    # cartesian indexing
    @test oa[CartesianIndex(3, 3, 3)] == oa[3, 3, 3]
  end

  @testset "Concatenating" begin
    # vector cat
    @test hcat(ov, ov) == OneHotMatrix(vcat(ov.indices, ov.indices), 10)
    @test hcat(ov, ov) isa OneHotMatrix
    @test vcat(ov, ov) == vcat(collect(ov), collect(ov))
    @test cat(ov, ov; dims = 3) == OneHotArray(cat(ov.indices, ov.indices; dims = 2), 10)

    # matrix cat
    @test hcat(om, om) == OneHotMatrix(vcat(om.indices, om.indices), 10)
    @test hcat(om, om) isa OneHotMatrix
    @test vcat(om, om) == vcat(collect(om), collect(om))
    @test cat(om, om; dims = 3) == OneHotArray(cat(om.indices, om.indices; dims = 2), 10)

    # array cat
    @test cat(oa, oa; dims = 3) == OneHotArray(cat(oa.indices, oa.indices; dims = 2), 10)
    @test cat(oa, oa; dims = 3) isa OneHotArray
    @test cat(oa, oa; dims = 1) == cat(collect(oa), collect(oa); dims = 1)

    # proper error handling of inconsistent sizes
    @test_throws DimensionMismatch hcat(ov, ov2)
    @test_throws DimensionMismatch hcat(om, om2)
  end

  @testset "Base.reshape" begin
    # reshape test
    @test reshape(oa, 10, 25) isa OneHotLike
    @test reshape(oa, 10, :) isa OneHotLike
    @test reshape(oa, :, 25) isa OneHotLike
    @test reshape(oa, 50, :) isa OneHotLike
    @test reshape(oa, 5, 10, 5) isa OneHotLike
    @test reshape(oa, (10, 25)) isa OneHotLike

    @testset "w/ cat" begin
      r = reshape(oa, 10, :)
      @test hcat(r, r) isa OneHotArray
      @test vcat(r, r) isa Array{Bool}
    end

    @testset "w/ argmax" begin
      r = reshape(oa, 10, :)
      @test argmax(r) == argmax(OneHotMatrix(reshape(oa.indices, :), 10))
      @test Flux._fast_argmax(r) == collect(reshape(oa.indices, :))
    end
  end

  @testset "Base.argmax" begin
    # argmax test
    @test argmax(ov) == argmax(convert(Array{Bool}, ov))
    @test argmax(om) == argmax(convert(Array{Bool}, om))
    @test argmax(om; dims = 1) == argmax(convert(Array{Bool}, om); dims = 1)
    @test argmax(om; dims = 2) == argmax(convert(Array{Bool}, om); dims = 2)
    @test argmax(oa; dims = 1) == argmax(convert(Array{Bool}, oa); dims = 1)
    @test argmax(oa; dims = 3) == argmax(convert(Array{Bool}, oa); dims = 3)
  end

  @testset "Forward map to broadcast" begin
    @test map(identity, oa) == oa
    @test map(x -> 2 * x, oa) == 2 .* oa
  end
end
