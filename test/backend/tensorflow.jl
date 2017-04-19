using TensorFlow
Flux.loadtf()

@testset "TensorFlow" begin

xs = rand(1, 20)
d = Affine(20, 10)

dt = tf(d)
@test d(xs) ≈ dt(xs)

@testset "Tensor interface" begin
  sess = TensorFlow.Session()
  X = placeholder(Float32)
  Y = Tensor(d, X)
  run(sess, initialize_all_variables())

  @test run(sess, Y, Dict(X=>Float32.(xs))) ≈ d(xs)
end

@testset "Stack Traces" begin
  model = TLP(Affine(10, 20), Affine(21, 15))
  dm = tf(model)
  e = try dm(rand(1, 10))
  catch e e end

  @test isa(e, DataFlow.Interpreter.Exception)
  @test e.trace[1].func == Symbol("Flux.Affine")
  @test e.trace[2].func == :TLP
end

end
