using Flux: throttle

@testset "Throttle" begin
  @testset "default behaviour" begin
    a = []
    f = throttle(()->push!(a, now()), 1, leading=true, trailing=false)
    f()
    f()
    f()
    sleep(1.01)
    @test length(a) == 1
  end

  @testset "leading behaviour" begin
    a = []
    f = throttle(()->push!(a, now()), 1, leading=true, trailing=false)
    f()
    @test length(a) == 1
    f()
    @test length(a) == 1
    sleep(1.01)
    f()
    @test length(a) == 2
  end

  @testset "trailing behaviour" begin
    a = []
    f = throttle(()->push!(a, now()), 1, leading=false, trailing=true)
    f()
    @test length(a) == 0
    f()
    @test length(a) == 0
    sleep(1.01)
    @test length(a) == 1
  end

  @testset "arguments" begin
    a = []
    f = throttle((x)->push!(a, x), 1, leading=true, trailing=true)
    f(1)
    @test a == [1]
    f(2)
    @test a == [1]
    f(3)
    @test a == [1]
    sleep(1.01)
    @test a == [1, 3]
  end
end

@testset "Jacobian" begin
  A = param(randn(2,2))
  x = randn(2)
  m(x) = A*x
  y = m(x)
  J = jacobian(m,x)
  @test J ≈ A.data
end
