using Test, Random
import Flux: activations

@testset "basic" begin
  @testset "helpers" begin
    @testset "activations" begin
      dummy_model = Chain(Dense(10,5,σ),Dense(5,2),softmax)
      x = rand(10)
      @test activations(Chain(), x) == []
      @test activations(dummy_model, x)[1] == dummy_model[1](x)
      @test activations(dummy_model, x)[2] == x |> dummy_model[1] |> dummy_model[2]
      @test activations(Chain(identity, x->:foo), x)[2] == :foo # results include `Any` type
    end
  end

  @testset "Chain" begin
    @test_nowarn Chain(Dense(10, 5, σ), Dense(5, 2))(randn(10))
    @test_throws DimensionMismatch Chain(Dense(10, 5, σ),Dense(2, 1))(randn(10))
    # numeric test should be put into testset of corresponding layer
  end

  @testset "Dense" begin
    @test  length(Dense(10, 5)(randn(10))) == 5
    @test_throws DimensionMismatch Dense(10, 5)(randn(1))
    @test_throws MethodError Dense(10, 5)(1) # avoid broadcasting
    @test_throws MethodError Dense(10, 5).(randn(10)) # avoid broadcasting

    @test Dense(10, 1, identity, initW = ones, initb = zeros)(ones(10,1)) == 10*ones(1, 1)
    @test Dense(10, 1, identity, initW = ones, initb = zeros)(ones(10,2)) == 10*ones(1, 2)
    @test Dense(10, 2, identity, initW = ones, initb = zeros)(ones(10,1)) == 10*ones(2, 1)
    @test Dense(10, 2, identity, initW = ones, initb = zeros)([ones(10,1) 2*ones(10,1)]) == [10 20; 10 20]

    @test size(Dense(10 => 5)(randn(10))) == (5,)
    @test size(Dense(10 => (5,))(randn(10))) == (5,)
    @test size(Dense(10 => (5,))(randn(10,7))) == (5,7)
    @test size(Dense(10 => (5,3))(randn(10,7))) == (5,3,7)
    @test size(Dense((10,7) => 5)(randn(10,7))) == (5,)

    @test Dense(10, (2,2), identity, initW=ones, initb=ones)(ones(10,3)) == 11 .* ones(2,2,3)
    @test Dense((3,3) => (2,2), sqrt, initW=ones, initb=zeros)(ones(3,3,5)) == 3 .* ones(2,2,5)
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
end
