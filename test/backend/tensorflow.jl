using TensorFlow
Flux.loadtf()

@testset "TensorFlow" begin

xs, ys = rand(1, 20), rand(1, 20)
d = Affine(20, 10)

dt = tf(d)
@test d(xs) ≈ dt(xs)

test_tupleio(tf)
test_recurrence(tf)
test_stacktrace(tf)
test_anon(tf)

@testset "Tensor interface" begin
  sess = TensorFlow.Session()
  X = placeholder(Float32)
  Y = Flux.TF.astensor(d, X)
  run(sess, global_variables_initializer())

  @test run(sess, Y, Dict(X=>xs)) ≈ d(xs)
end

@testset "Ops" begin
  A = randn(Float32,(5,5))
  u,s,v = tf(@net x -> svd(x))(A)
  @test A ≈ u*diagm(s)*transpose(v)
  @test tf(@net x -> inv(x))(A) ≈ inv(A)
  @test tf(@net x -> det(x))(A) ≈ det(A)
  A = randn(Float32,(6,3))
  @test tf(@net x -> transpose(x))(A) ≈ transpose(A)
  A = randn(Float32,(6,3,2))
  @test tf(@net (x,y) -> permutedims(x,y))(A,[3,2,1]) ≈ permutedims(A,[3,2,1])
  A1 = randn(Float32,(4,1))
  A2 = randn(Float32,(4,1))
  @test tf(@net (x,y) -> cat(2,x,y))(A1,A2) ≈ cat(2,A1,A2)
  @test tf(@net x -> length(x))(A1) == length(A1)
  A = randn(Float32,(5,5))
  @test tf(@net x -> diag(x))(A) ≈ diag(A)
  A = randn(Float32,(5,))
  @test tf(@net x -> diagm(x))(A) ≈ diagm(A)
  A = randn(4,5)
  @test tf(@net x -> size(x))(A) == [4,5]
  @test tf(@net (x,y) -> size(x,y))(A,1) == 4
  A = randn(6,5)
  A = A'*A
  @test tf(@net x -> chol(x))(A) ≈ chol(A)
  A = randn(Float32,(6,3))
  @test transpose(tf(@net (x,y) -> reshape(x,y))(transpose(A),[2,9])) ≈ reshape(A,(9,2)) # Note: TF is row major and julia is not
  A = randn(Float32,(4,3,1))
  @test tf(@net (x,y) -> Flux.tile(x,y))(A,[1,1,3]) ≈ repeat(A,outer=(1,1,3))
  @test tf(@net (x,y) -> fill(x,y))(3.2,[3,2]) ≈ convert(Array{Float32},3.2*ones(3,2))
  @test typeof(tf(@net x -> Flux.cast(x,Int32))(A)) == Array{Int32,3}
  A = randn(Float32,(5,5))
  b = randn(Float32,(5,1))
  @test tf(@net (x,y) -> solve(x,y))(A,b) ≈ A\b
  _,A,_ = lu(A)
  @test tf(@net (x,y) -> triangular_solve(x,y))(A,b) ≈ A\b
  @test size(tf(@net x -> randu(x))([2,3])) == (2,3)
  @test size(tf(@net x -> randn(x))([2,3])) == (2,3)
  m = tf(@net (x,y) -> Flux.expand_dims(x,y))
  A = randn(Float32,(3,2))
  @test m(A,1) ≈ Flux.expand_dims(A,1)
  @test m(A,2) ≈ Flux.expand_dims(A,2)
  @test m(A,3) ≈ Flux.expand_dims(A,3)
end

end
