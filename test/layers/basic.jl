using Test, Random
import Flux: activations

@testset "basic" begin
  @testset "helpers" begin
    @testset "activations" begin
      dummy_model = Chain(x->x.^2, x->x .- 3, x -> tan.(x))
      x = randn(10)
      @test activations(dummy_model, x)[1] == x.^2
      @test activations(dummy_model, x)[2] == (x.^2 .- 3)
      @test activations(dummy_model, x)[3] == tan.(x.^2 .- 3)

      @test activations(Chain(), x) == ()
      @test activations(Chain(identity, x->:foo), x)[2] == :foo # results include `Any` type
    end
  end

  @testset "Chain" begin
    @test_nowarn Chain(Dense(10, 5, σ), Dense(5, 2))(randn(10))
    @test_throws DimensionMismatch Chain(Dense(10, 5, σ),Dense(2, 1))(randn(10))
    # numeric test should be put into testset of corresponding layer
  end

  @testset "Activations" begin
    c = Chain(Dense(3,5,relu), Dense(5,1,relu))
    X = Float32.([1.0; 1.0; 1.0])
    @test_nowarn gradient(()->Flux.activations(c, X)[2][1], params(c))
  end

  @testset "Dense" begin
    @test length(Dense(10, 5)(randn(10))) == 5
    @test_throws DimensionMismatch Dense(10, 5)(randn(1))
    @test_throws MethodError Dense(10, 5)(1) # avoid broadcasting
    @test_throws MethodError Dense(10, 5).(randn(10)) # avoid broadcasting

    @test Dense(10, 1, identity, initW = ones, initb = zeros)(ones(10,1)) == 10*ones(1, 1)
    @test Dense(10, 1, identity, initW = ones, initb = zeros)(ones(10,2)) == 10*ones(1, 2)
    @test Dense(10, 2, identity, initW = ones, initb = zeros)(ones(10,1)) == 10*ones(2, 1)
    @test Dense(10, 2, identity, initW = ones, initb = zeros)([ones(10,1) 2*ones(10,1)]) == [10 20; 10 20]

  end

  @testset "Diagonal" begin
    @test length(Flux.Diagonal(10)(randn(10))) == 10
    @test length(Flux.Diagonal(10)(1)) == 10
    @test length(Flux.Diagonal(10)(randn(1))) == 10
    @test_throws DimensionMismatch Flux.Diagonal(10)(randn(2))

    @test Flux.Diagonal(2)([1 2]) == [1 2; 1 2]
    @test Flux.Diagonal(2)([1,2]) == [1,2]
    @test Flux.Diagonal(2)([1 2; 3 4]) == [1 2; 3 4]
  end

  @testset "Maxout" begin
    # Note that the normal common usage of Maxout is as per the docstring
    # These are abnormal constructors used for testing purposes

    @testset "Constructor" begin
      mo = Maxout(() -> identity, 4)
      input = rand(40)
      @test mo(input) == input
    end

    @testset "simple alternatives" begin
      mo = Maxout((x -> x, x -> 2x, x -> 0.5x))
      input = rand(40)
      @test mo(input) == 2*input
    end

    @testset "complex alternatives" begin
      mo = Maxout((x -> [0.5; 0.1]*x, x -> [0.2; 0.7]*x))
      input = [3.0 2.0]
      target = [0.5, 0.7].*input
      @test mo(input) == target
    end

    @testset "params" begin
      mo = Maxout(()->Dense(32, 64), 4)
      ps = params(mo)
      @test length(ps) == 8  #4 alts, each with weight and bias
    end
  end

  @testset "SkipConnection" begin
    @testset "zero sum" begin
      input = randn(10, 10, 10, 10)
      @test SkipConnection(x -> zeros(size(x)), (a,b) -> a + b)(input) == input
    end

    @testset "concat size" begin
      input = randn(10, 2)
      @test size(SkipConnection(Dense(10,10), (a,b) -> cat(a, b, dims = 2))(input)) == (10,4)
    end
  end

  @testset "GroupedConvolutions" begin
    input256 = randn(7, 7, 256, 16)

    @testset "constructor" begin
      path1 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      path2 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      path3 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      path4 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )

      # the number of paths is not greater than 1
      @test_throws ErrorException GroupedConvolutions(+)
      @test_throws ErrorException GroupedConvolutions(+, split=false)
      @test_throws ErrorException GroupedConvolutions(+, split=true)
      @test_throws ErrorException GroupedConvolutions(+, path1)
      @test_throws ErrorException GroupedConvolutions(+, path1, split=false)
      @test_throws ErrorException GroupedConvolutions(+, path1, split=true)

      # varargs
      group3 = GroupedConvolutions(+, path1, path2, path3, split=true)
      @test size(group3.paths, 1) == 3
      @test group3.split == true
      group4 = GroupedConvolutions(+, path1, path2, path3, path4, split=true)
      @test size(group4.paths, 1) == 4
      @test group4.split == true
    end

    @testset "sum split" begin
      path1 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      path2 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      path3 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      path4 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      result1 = path1(input256[:,:,1:64,:])
      result2 = path2(input256[:,:,65:128,:])
      result3 = path3(input256[:,:,129:192,:])
      result4 = path4(input256[:,:,193:256,:])
      group3 = GroupedConvolutions(+, path1, path2, path3, split=true)
      group4 = GroupedConvolutions(+, path1, path2, path3, path4, split=true)

      # summation for 3 paths
      # the number of feature maps in the input (256) is not divisible by the number of paths of the GroupedConvolution (3)
      @test_throws ErrorException group3(input256)

      # summation for 4 paths
      result = group4(input256)
      @test size(result) == size(input256)
      @test result == result1 + result2 + result3 + result4
    end

    @testset "sum no split" begin
      path1 = Chain(
                Conv((1,1), 256 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      path2 = Chain(
                Conv((1,1), 256 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      path3 = Chain(
                Conv((1,1), 256 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      path4 = Chain(
                Conv((1,1), 256 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
              )
      result1 = path1(input256)
      result2 = path2(input256)
      result3 = path3(input256)
      result4 = path4(input256)
      group3 = GroupedConvolutions(+, path1, path2, path3)
      group4 = GroupedConvolutions(+, path1, path2, path3, path4)

      # summation for 3 paths
      # does not throw exception anymore
      result = group3(input256)
      @test size(result) == size(input256)
      @test result == result1 + result2 + result3

      # summation for 4 paths
      result = group4(input256)
      @test size(result) == size(input256)
      @test result == result1 + result2 + result3 + result4
    end

    @testset "cat split" begin
      path1 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1))
              )
      path2 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1))
              )
      path3 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1))
              )
      path4 = Chain(
                Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1))
              )
      result1 = path1(input256[:,:,1:64,:])
      result2 = path2(input256[:,:,65:128,:])
      result3 = path3(input256[:,:,129:192,:])
      result4 = path4(input256[:,:,193:256,:])
      group3 = GroupedConvolutions((a,b,c) -> cat(a, b, c, dims=3), path1, path2, path3, split=true)
      group4 = GroupedConvolutions((a,b,c,d) -> cat(a, b, c, d, dims=3), path1, path2, path3, path4, split=true)
      result = group4(input256)

      # concatenation for 3 paths
      # the number of feature maps in the input (256) is not divisible by the number of paths of the GroupedConvolution (3)
      @test_throws ErrorException group3(input256)

      # concatenation for 4 paths
      @test size(result) == (7, 7, 4*4, 16)
      @test result == cat(result1, result2, result3, result4, dims=3)
    end

    @testset "cat no split" begin
      path1 = Chain(
                Conv((1,1), 256 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1))
              )
      path2 = Chain(
                Conv((1,1), 256 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1))
              )
      path3 = Chain(
                Conv((1,1), 256 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1))
              )
      path4 = Chain(
                Conv((1,1), 256 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1))
              )
      result1 = path1(input256)
      result2 = path2(input256)
      result3 = path3(input256)
      result4 = path4(input256)
      group3 = GroupedConvolutions((a,b,c) -> cat(a, b, c, dims=3), path1, path2, path3)
      group4 = GroupedConvolutions((a,b,c,d) -> cat(a, b, c, d, dims=3), path1, path2, path3, path4)

      # concatenation for 3 paths
      # does not throw exception anymore
      result = group3(input256)
      @test size(result) == (7, 7, 3*4, 16)
      @test result == cat(result1, result2, result3, dims=3)

      # concatenation for 4 paths
      result = group4(input256)
      @test size(result) == (7, 7, 4*4, 16)
      @test result == cat(result1, result2, result3, result4, dims=3)
    end

    @testset "mixed paths" begin
      path1 = Conv((1,1), 128=>64, pad=(0, 0), stride=(1, 1))
      path2 = Chain(
                Conv((1,1), 128 => 4, pad=(0, 0), stride=(1, 1)),
                Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
                Conv((1,1), 4 => 64, pad=(0, 0), stride=(1, 1))
              )
      result1 = path1(input256[:,:,1:128,:])
      result2 = path2(input256[:,:,129:256,:])
      group2 = GroupedConvolutions(+, path1, path2, split=true)
      result = group2(input256)

      # summation for 2 different paths
      @test size(result) == (7, 7, 64, 16)
      @test result == result1 + result2
    end
  end

  @testset "ChannelShuffle" begin
    @testset "constructor" begin
      # the number of groups is not greater than 1
      @test_throws ErrorException ChannelShuffle(-1)
      @test_throws ErrorException ChannelShuffle(0)
      @test_throws ErrorException ChannelShuffle(1)
    end

    @testset "channel shuffling" begin
      input3 = reshape(collect(1:1*1*3*1),(1,1,3,1))
      input4 = reshape(collect(1:1*1*4*1),(1,1,4,1))
      input8 = reshape(collect(1:1*1*8*1),(1,1,8,1))
      input9 = reshape(collect(1:1*1*9*1),(1,1,9,1))
      input16 = reshape(collect(1:1*1*16*1),(1,1,16,1))
      input256 = reshape(collect(1:7*7*256*16),(7,7,256,16))
      shuffle2 = ChannelShuffle(2)
      shuffle3 = ChannelShuffle(3)
      shuffle4 = ChannelShuffle(4)
      shuffle8 = ChannelShuffle(8)

      # the number of feature maps in the input is not divisible by the square of the number of groups of the ChannelShuffle
      @test_throws ErrorException shuffle3(input3)
      @test_throws ErrorException shuffle3(input4)
      @test_throws ErrorException shuffle4(input8)

      # ab,cd               -> ac,bd               (2 groups)
      # 12,34               -> 13,24               (2 groups)
      @test shuffle2(input4)[1,1,:,1] == [1, 3, 2, 4]

      # abcd,efgh           -> aebf,cgdh           (2 groups)
      # 12434,5678          -> 1526,3748           (2 groups)
      @test shuffle2(input8)[1,1,:,1] == [1, 5, 2, 6, 3, 7, 4, 8]

      # abcdefgh,ijklmnop   -> aibjckdl,emfngohp   (2 groups)
      # 12345678,9...       -> 192.3.4.,5.6.7.8.   (2 groups)
      @test shuffle2(input16)[1,1,:,1] == [1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16]

      # abc,def,ghi         -> adg,beh,cfi         (3 groups)
      # 123,456,789         -> 147,258,369         (3 groups)
      @test shuffle3(input9)[1,1,:,1] == [1, 4, 7, 2, 5, 8, 3, 6, 9]

      # abcd,efgh,ijkl,mnop -> aeim,bfjn,cgko,dhlp (4 groups)
      # 1234,5678,9...      -> 159.,26..,37..,48.. (4 groups)
      @test shuffle4(input16)[1,1,:,1] == [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]

      # bigger arrays
      @test size(shuffle8(input256)) == size(input256)
    end
  end

  @testset "ShuffledGroupedConvolutions" begin
    input256 = randn(7, 7, 256, 16)
    path1 = Chain(
              Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
              Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
              Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
            )
    path2 = Chain(
              Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
              Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
              Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
            )
    path3 = Chain(
              Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
              Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
              Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
            )
    path4 = Chain(
              Conv((1,1), 64 => 4, pad=(0, 0), stride=(1, 1)),
              Conv((3,3), 4 => 4, pad=(1, 1), stride=(1, 1)),
              Conv((1,1), 4 => 256, pad=(0, 0), stride=(1, 1))
            )
    group4 = GroupedConvolutions(+, path1, path2, path3, path4, split=true)
    result1 = path1(input256[:,:,1:64,:])
    result2 = path2(input256[:,:,65:128,:])
    result3 = path3(input256[:,:,129:192,:])
    result4 = path4(input256[:,:,193:256,:])
    shuffle3 = ChannelShuffle(3)
    shuffle4 = ChannelShuffle(4)

    @testset "constructor" begin
      # the number of groups in the ChannelShuffle layer (3) is not equal to the number of paths in the GroupedConvolutions (4)
      @test_throws ErrorException ShuffledGroupedConvolutions(group4, shuffle3)

      # varargs
      shuffled_group3 = ShuffledGroupedConvolutions(+, path1, path2, path3, split=true)
      @test size(shuffled_group3.group.paths, 1) == 3
      @test shuffled_group3.group.split == true
      @test shuffled_group3.shuffle.ngroups == 3
      shuffled_group4 = ShuffledGroupedConvolutions(+, path1, path2, path3, path4, split=true)
      @test size(shuffled_group4.group.paths, 1) == 4
      @test shuffled_group4.group.split == true
      @test shuffled_group4.shuffle.ngroups == 4
    end

    @testset "shuffled grouped convolutions" begin
      shuffle_group4 = ShuffledGroupedConvolutions(group4, shuffle4)
      result = shuffle_group4(input256)
      @test size(result) == size(input256)
      @test result == shuffle4(group4(input256))
    end
  end
end
