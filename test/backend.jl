xs = rand(20)
d = Affine(20, 10)

# MXNet

@mxonly let dm = mxnet(d, (20, 1))
  @test d(xs) ≈ dm(xs)
end

@mxonly let
  # TODO: test run
  using MXNet
  f = mx.FeedForward(Chain(d, softmax))
  @test mx.infer_shape(f.arch, data = (20, 1))[2] == [(10, 1)]

  m = Chain(Input(28,28), Conv2D((5,5), out = 3), MaxPool((2,2)),
            flatten, Affine(1587, 10), softmax)
  f = mx.FeedForward(m)
  @test mx.infer_shape(f.arch, data = (20, 20, 5, 1))[2] == [(10, 1)]
end

@mxonly let
  model = TLP(Affine(10, 20), Affine(21, 15))
  info("The following warning is normal")
  e = try mxnet(model, (10, 1))
  catch e e end

  @test e.trace[1].func == Symbol("Flux.Affine")
  @test e.trace[2].func == :TLP
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
