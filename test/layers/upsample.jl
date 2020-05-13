using Flux: BilinearUpsample2d
using Test

@testset "BilinearUpsample2d" begin
  @test size(BilinearUpsample2d((2, 2))(rand(2, 2, 1, 1))) == (4, 4, 1, 1)
  @test size(BilinearUpsample2d((3, 3))(rand(2, 2, 1, 1))) == (6, 6, 1, 1)
  @test size(BilinearUpsample2d((2, 3))(rand(2, 2, 10, 10))) == (4, 6, 10, 10)
  @test size(BilinearUpsample2d((3, 2))(rand(2, 2, 10, 10))) == (6, 4, 10, 10)

  @test_throws MethodError BilinearUpsample2d((2, 2))(rand(2, 2))

  @test BilinearUpsample2d((3, 2))([1. 2.; 3. 4.][:,:,:,:]) ≈
   [1//1  5//4    7//4    2//1;
    1//1  5//4    7//4    2//1;
    5//3  23//12  29//12  8//3;
    7//3  31//12  37//12  10//3;
    3//1  13//4   15//4   4//1;
    3//1  13//4   15//4   4//1][:,:,:,:]

    testimg1 = [1. 0.; 0 0][:,:,:,:]
    factors1 = (3, 2)
    f1(x) = sum(BilinearUpsample2d(factors1)(x))
    df1(x) = Flux.gradient(f1, x)[1]
    @test df1(testimg1) ≈ fill(eltype(testimg1).(prod(factors1)), size(testimg1))

    testimg2 = [1. 0.; 0 0][:,:,:,:]
    factors2 = (3, 2)
    f2(x) = BilinearUpsample2d(factors2)(x)[3,2]
    df2(x) = Flux.gradient(f2, x)[1]
    @test df2(testimg2) ≈
    [1//2  1//6
     1//4  1//12][:,:,:,:]
end
