using Flux: throttle, initn, glorot_uniform, glorot_normal, jacobian

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

@testset "Initialization" begin
  # Set random seed so that these tests don't fail randomly
  srand(0)
  # initn() should yield a kernel with stddev ~= 1e-2
  v = initn(10, 10)
  @test std(v) > 0.9*1e-2
  @test std(v) < 1.1*1e-2

  # glorot_uniform should yield a kernel with stddev ~= sqrt(6/(n_in + n_out)),
  # and glorot_normal should yield a kernel with stddev != 2/(n_in _ n_out)
  for (n_in, n_out) in [(100, 100), (100, 400)]
    v = glorot_uniform(n_in, n_out)
    @test minimum(v) > -1.1*sqrt(6/(n_in + n_out))
    @test minimum(v) < -0.9*sqrt(6/(n_in + n_out))
    @test maximum(v) >  0.9*sqrt(6/(n_in + n_out))
    @test maximum(v) <  1.1*sqrt(6/(n_in + n_out))

    v = glorot_normal(n_in, n_out)
    @test std(v) > 0.9*sqrt(2/(n_in + n_out))
    @test std(v) < 1.1*sqrt(2/(n_in + n_out))
  end
end

@testset "Params" begin
  m = Dense(10, 5)
  @test size.(params(m)) == [(5, 10), (5,)]
  m = RNN(10, 5)
  @test size.(params(m)) == [(5, 10), (5, 5), (5,), (5,)]
end

@testset "batches" begin
  x = rand(10)
  b = batches(x) |> collect
  @test length(b) == 10
  @test all(i -> b[i] == x[i:i],1:10)

  b = batches(x, batchsize=3) |> collect
  @test length(b) == 4
  @test b[1] == x[1:3]
  @test b[2] == x[4:6]
  @test b[3] == x[7:9]
  @test b[4] == x[10:10]

  y = rand(8)
  @test_throws AssertionError batches(x, y)

  x = reshape(x, 2, 5)
  y = rand(5)
  b = batches(x,y, batchsize=2) |> collect
  @test length(b) == 3
  @test b[1] == (x[:,1:2], y[1:2])
  @test b[2] == (x[:,3:4], y[3:4])
  @test b[3] == (x[:,5:5], y[5:5])
end

@testset "mat" begin
  x = rand(2)
  @test mat(x) == reshape(x, 1, 2)
  x = rand(2,2)
  @test mat(x) == x
  x = rand(2,2,2)
  @test mat(x) == reshape(x,4,2)
  x = rand(2,2,2,2)
  @test mat(x) == reshape(x,8,2)
end
