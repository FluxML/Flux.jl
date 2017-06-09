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
end

end
