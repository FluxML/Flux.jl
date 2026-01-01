
@testset "Tracker.jl" begin
  @testset "some simple models" begin
    m1 =  Dense(ones32(2,3), fill(0.1f0,2), abs2)
    x1 = Float32[1,2,3]
    (_, v1), g1 = withgradient(m1, x1) do m, x
      y1 = m(x)
      sum(abs2, y1 .- [4, 5]), y1
    end
    @test v1 ≈ m1(x1)
    g1z = gradient(m1, x1) do m, x
      sum(abs2, m(x) .- [4, 5])
    end
    @test g1[1].weight ≈ g1z[1].weight
    @test g1[1].bias ≈ g1z[1].bias

    m2 = Chain(Conv((2,2), 3 => 1, relu), Flux.flatten, Dense(20 => 1, tanh), only)
    x2 = randn32(5,6,3,1)
    v2, g2 = withgradient(m -> m(x2), m2)
    g2z = gradient(m -> m(x2), m2)
    @test g2[1].layers[1].weight ≈ g2z[1].layers[1].weight
    @test g2[1].layers[1].bias ≈ g2z[1].layers[1].bias
    @test g2[1].layers[3].weight ≈ g2z[1].layers[3].weight
  end

  @testset "Dropout" begin
    g1z = gradient(sum∘Dropout(0.5), ones(1000))
    v1, g1 = withgradient(sum∘Dropout(0.5), ones(1000))
    @test 800<v1<1200
    @test sum(g1[1]) ≈ v1
    @test 400 < count(iszero, g1[1]) < 600
  end
end

