xs = rand(20)
d = Affine(20, 10)

let dt = tf(d)
  @test d(xs) ≈ dt(xs)
end

# TensorFlow native integration

using TensorFlow

let
  sess = TensorFlow.Session()
  X = placeholder(Float32)
  Y = Tensor(d, X)
  run(sess, initialize_all_variables())

  @test run(sess, Y, Dict(X=>xs')) ≈ d(xs)'
end
