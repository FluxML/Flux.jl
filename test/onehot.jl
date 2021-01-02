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
  @test y[:,1] isa Flux.OneHot
  @test y[:,:] isa Flux.OneHotArray
end

@testset "abstractmatrix onehotvector multiplication" begin
  A = [1 3 5; 2 4 6; 3 6 9]
  b1 = Flux.OneHot(3, 1)
  b2 = Flux.OneHot(5, 3)

  @test A*b1 == A[:,1]
  @test_throws DimensionMismatch A*b2
end

@testset "OneHot primitive" begin
  using Flux: OneHot
  idx = rand(5:10)
  ot = OneHot(10)
  o = OneHot{10}(idx)

  @test Flux.onehotsize(o) == 10
  @test o == OneHot(10, idx)
  @test o == ot(idx)

  @test size(o) == (10,)
  for i = 1:10
    @test o[i] == (i == idx)
  end
  @test_throws BoundsError o[11]

  @test UInt32(o) == UInt32(idx)
  @test UInt64(o) == UInt64(idx)
  @test Int32(o) == Int32(idx)
  @test Int64(o) == Int64(idx)

  @test convert(UInt32, o) == UInt32(idx)
  @test convert(UInt64, o) == UInt64(idx)
  @test convert(Int32, o) == Int32(idx)
  @test convert(Int64, o) == Int64(idx)
  @test convert(Int8, o) == Int8(idx)
  @test_throws InexactError convert(Bool, o)
  @test convert(ot, idx) == o
  @test convert(ot, UInt8(idx)) == o
  @test convert(ot, UInt16(idx)) == o
  @test convert(ot, UInt32(idx)) == o
  @test convert(ot, UInt64(idx)) == o
  @test convert(ot, UInt128(idx)) == o
  @test convert(ot, Int8(idx)) == o
  @test convert(ot, Int16(idx)) == o
  @test convert(ot, Int32(idx)) == o
  @test convert(ot, Int64(idx)) == o
  @test convert(ot, Int128(idx)) == o

  @test convert(ot, true) == OneHot(10, 1)
  @test convert(ot, false) == OneHot(10, 0)

  @test convert(OneHot(15), o) == OneHot(15, idx)
  @test convert(OneHot(idx+1), o) == OneHot(idx+1, idx)
  @test_throws Flux.OneHotEncodeError convert(OneHot(idx-1), o)

  @test zero(o) == OneHot(10, 0)
  @test !iszero(o)
  @test iszero(zero(o))

  @test_throws Flux.OneHotEncodeError convert(ot, -1)
  @test_throws Flux.OneHotEncodeError convert(ot, 12)
  @test_throws Flux.OneHotEncodeError convert(ot, typemax(Int64))
  @test_throws Flux.OneHotEncodeError convert(ot, typemax(UInt64))
end

@testset "OneHotArray" begin
  using Flux: OneHotArray
  idx = rand(5:10)
  ot = OneHot(10)
  o = OneHot{10}(idx)
  ind = rand(1:10, 5, 5)
  oa = OneHotArray{10}(ind)

  @test Flux.onehotsize(oa) == 10
  @test OneHotArray(15, oa) |> Flux.onehotsize == 15

  @test size(oa) == (10, 5, 5)
  @test oa[:, 5, 5] == oa.onehots[5, 5]
  @test oa[:, 5, :] == OneHotArray(oa.onehots[5, :])
  @test oa[ind[1], 1, 1]

  @test vcat(o, o) == vcat(collect(o), collect(o))
  @test hcat(o, o).onehots == [o,o]
  @test cat(o, o; dims=1) == vcat(collect(o), collect(o))
  @test cat(o, o; dims=2).onehots == [o,o]

  @test vcat(oa, oa) == vcat(collect(oa), collect(oa))
  @test hcat(oa, oa).onehots == vcat(oa.onehots, oa.onehots)
  @test cat(oa, oa; dims=1) == vcat(collect(oa), collect(oa))
  @test cat(oa, oa; dims=2).onehots == vcat(oa.onehots, oa.onehots)
  @test cat(oa, oa; dims=3).onehots == cat(oa.onehots, oa.onehots; dims=2)

  @test reshape(oa, 10, 25) isa OneHotArray
  @test reshape(oa, 10, :) isa OneHotArray
  @test reshape(oa, :, 25) isa OneHotArray
  @test reshape(oa, 50, :) isa Base.ReshapedArray{Bool, 2}
  @test reshape(oa, 5, 10, 5) isa Base.ReshapedArray{Bool, 3}

end
