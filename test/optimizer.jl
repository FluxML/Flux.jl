@testset "training julia models" begin

  @testset "linear regression" begin
    srand(0)

    truth = Float32[0, 4, 2, 2, -3, 6, -1, 3, 2, -5]'

    data = map(1:256) do i
      x = rand(Float32, 10)
      x, truth * x + 3rand(Float32)
    end

    # It's hard to tell if an optimizer works exactly right, but we
    # can at least ensure that they all converge at the right point
    for opt in (SGD(), SGD(momentum=.9, decay=.01), AdaGrad(lr=1.), RMSProp(lr=.1), AdaDelta(lr=1e3), Adam())
      model = Affine(10, 1)

      Flux.train!(model, data, epoch=10, opt=opt)

      @test cor(reshape.((model.W.x, truth), 10)...) > .99
    end
  end

  @testset "logistic regression" begin
    srand(0)

    truth = Float32[0, 4, 2, 2, -3, 6, -1, 3, 2, -5]'

    data = map(1:256) do i
      x = rand(Float32, 10)
      x, truth * x + 2rand(Float32) > 5f0
    end

    for opt in (SGD(), SGD(momentum=.9, decay=.01), AdaGrad(lr=1.), RMSProp(lr=.1), AdaDelta(lr=1e3), Adam())
      model = Chain(Affine(10, 1), σ)

      Flux.train!(model, data, epoch=10)

      @test cor(reshape.((model.layers[1].W.x, truth), 10)...) > .99
    end
  end

end

