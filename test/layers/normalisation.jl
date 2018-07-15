using Flux: testmode!
using Flux.Tracker: data

@testset "Dropout" begin
  x = [1.,2.,3.]
  @test x == testmode!(Dropout(0.1))(x)
  @test x == Dropout(0)(x)
  @test zeros(x) == Dropout(1)(x)

  x = rand(100)
  m = Dropout(0.9)
  y = m(x)
  @test count(a->a==0, y) > 50
  testmode!(m)
  y = m(x)
  @test count(a->a==0, y) == 0
  testmode!(m, false)
  y = m(x)
  @test count(a->a==0, y) > 50

  x = rand(100)
  m = Chain(Dense(100,100),
            Dropout(0.9))
  y = m(x)
  @test count(a->a == 0, y) > 50
  testmode!(m)
  y = m(x)
  @test count(a->a == 0, y) == 0
end

@testset "BatchNorm" begin
  let m = BatchNorm(2), x = param([1 3 5;
                                   2 4 6])

    @test m.β.data == [0, 0]  # initβ(2)
    @test m.γ.data == [1, 1]  # initγ(2)
    # initial m.σ is 1
    # initial m.μ is 0
    @test m.active

    # @test m(x).data ≈ [-1 -1; 0 0; 1 1]'
    m(x)

    # julia> x
    #  2×3 Array{Float64,2}:
    #  1.0  3.0  5.0
    #  2.0  4.0  6.0
    #
    # μ of batch will be
    #  (1. + 3. + 5.) / 3 = 3
    #  (2. + 4. + 6.) / 3 = 4
    #
    # ∴ update rule with momentum:
    #  .1 * 3 + 0 = .3
    #  .1 * 4 + 0 = .4
    @test m.μ ≈ reshape([0.3, 0.4], 2, 1)

    @test m.σ² ≈ 0.1 .* var(x.data, 2, corrected=false)*3/2  + 0.9 .* [1., 1.]

    testmode!(m)
    @test !m.active

    y = m(x).data
    @test y ≈ data((x .- m.μ) ./ sqrt.(m.σ² .+ m.ϵ))
  end

  # with activation function
  let m = BatchNorm(2, sigmoid), x = param([1 3 5;
                                            2 4 6])
    @test m.active
    m(x)

    testmode!(m)
    @test !m.active

    y = m(x).data
    @test y ≈ data(sigmoid.((x .- m.μ) ./ sqrt.(m.σ² .+ m.ϵ)))
  end

  let m = BatchNorm(2), x = param(reshape(1:6, 3, 2, 1))
    y = reshape(permutedims(x, [2, 1, 3]), 2, :)
    y = permutedims(reshape(m(y), 2, 3, 1), [2, 1, 3])
    @test m(x) == y
  end

  let m = BatchNorm(2), x = param(reshape(1:12, 2, 3, 2, 1))
    y = reshape(permutedims(x, [3, 1, 2, 4]), 2, :)
    y = permutedims(reshape(m(y), 2, 2, 3, 1), [2, 3, 1, 4])
    @test m(x) == y
  end

  let m = BatchNorm(2), x = param(reshape(1:24, 2, 2, 3, 2, 1))
    y = reshape(permutedims(x, [4, 1, 2, 3, 5]), 2, :)
    y = permutedims(reshape(m(y), 2, 2, 2, 3, 1), [2, 3, 4, 1, 5])
    @test m(x) == y
  end
end
