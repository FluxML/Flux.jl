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

  @testset "svd" begin
    A = convert(Array{Float32},randn(5,5))
    @net f(x) = svd(x)
    m = tf(f)
    u,s,v = m(A)
    @test A ≈ u*diagm(s)*transpose(v)
  end

  @testset "inv" begin
    @net f(x) = inv(x)
    m = tf(f)
    @test m(A) ≈ inv(A)
  end

  @testset "det" begin
    @net f(x) = det(x)
    m = tf(f)
    @test m(A) ≈ det(A)
  end

end

end
