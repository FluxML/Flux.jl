xs = rand(20)
d = Affine(20, 10)

# MXNet

@mxonly let dm = mxnet(d, (1, 20))
  @test d(xs) ≈ dm(xs)
end

@mxonly let
  using MXNet
  f = mx.FeedForward(Chain(d, softmax))
  @test isa(f, mx.FeedForward)
  # TODO: test run
end

# TensorFlow

@tfonly let dt = tf(d)
  @test d(xs) ≈ dt(xs)
end

@tfonly let
  using TensorFlow

  sess = TensorFlow.Session()
  X = placeholder(Float32)
  Y = Tensor(d, X)
  run(sess, initialize_all_variables())

  @test run(sess, Y, Dict(X=>xs')) ≈ d(xs)'
end
