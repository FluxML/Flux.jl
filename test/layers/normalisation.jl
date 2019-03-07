using Flux: testmode!
using Flux.Tracker: data

@testset "Dropout" begin
  x = [1.,2.,3.]
  @test x == testmode!(Dropout(0.1))(x)
  @test x == Dropout(0)(x)
  @test zero(x) == Dropout(1)(x)

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

    # julia> .1 .* var(x, dims = 2, corrected=false) .* (3 / 2).+ .9 .* [1., 1.]
    # 2×1 Array{Float64,2}:
    #  1.3
    #  1.3
    @test m.σ² ≈ .1 .* var(x.data, dims = 2, corrected=false) .* (3 / 2).+ .9 .* [1., 1.]

    testmode!(m)
    @test !m.active

    x′ = m(x).data
    @test isapprox(x′[1], (1 .- 0.3) / sqrt(1.3), atol = 1.0e-5)
  end

  # with activation function
  let m = BatchNorm(2, sigmoid), x = param([1 3 5;
                                            2 4 6])
    @test m.active
    m(x)

    testmode!(m)
    @test !m.active

    y = m(x).data
    @test isapprox(y, data(sigmoid.((x .- m.μ) ./ sqrt.(m.σ² .+ m.ϵ))), atol = 1.0e-7)
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

  let m = BatchNorm(32), x = randn(Float32, 416, 416, 32, 1);
    m(x)
    @test (@allocated m(x)) <  100_000_000
  end
end


@testset "InstanceNorm" begin
  # helper functions
  expand_inst = (x, as) -> reshape(repeat(x, outer=[1, as[length(as)]]), as...)
  # begin tests
  let m = InstanceNorm(2), sizes = (3, 2, 2),
      x = param(reshape(collect(1:prod(sizes)), sizes))

      @test m.β.data == [0, 0]  # initβ(2)
      @test m.γ.data == [1, 1]  # initγ(2)

      @test m.active

      m(x)

      #julia> x
      #[:, :, 1] =
      # 1.0  4.0
      # 2.0  5.0
      # 3.0  6.0
      #
      #[:, :, 2] =
      # 7.0  10.0
      # 8.0  11.0
      # 9.0  12.0
      #
      # μ will be
      # (1. + 2. + 3.) / 3 = 2.
      # (4. + 5. + 6.) / 3 = 5.
      #
      # (7. + 8. + 9.) / 3 = 8.
      # (10. + 11. + 12.) / 3 = 11.
      #
      # ∴ update rule with momentum:
      # (1. - .1) * 0 + .1 * (2. + 8.) / 2 = .5
      # (1. - .1) * 0 + .1 * (5. + 11.) / 2 = .8
      @test m.μ ≈ [0.5, 0.8]
      # momentum * var * num_items / (num_items - 1) + (1 - momentum) * sigma_sq
      # julia> reshape(mean(.1 .* var(x.data, dims = 1, corrected=false) .* (3 / 2), dims=3), :) .+ .9 .* 1.
      # 2-element Array{Float64,1}:
      #  1.
      #  1.
      @test m.σ² ≈ reshape(mean(.1 .* var(x.data, dims = 1, corrected=false) .* (3 / 2), dims=3), :) .+ .9 .* 1.

      testmode!(m)
      @test !m.active

      x′ = m(x).data
      @test isapprox(x′[1], (1 - 0.5) / sqrt(1. + 1f-5), atol = 1.0e-5)
  end
  # with activation function
  let m = InstanceNorm(2, sigmoid), sizes = (3, 2, 2),
      x = param(reshape(collect(1:prod(sizes)), sizes))

    affine_shape = collect(sizes)
    affine_shape[1] = 1

    @test m.active
    m(x)

    testmode!(m)
    @test !m.active

    y = m(x).data
    @test isapprox(y, data(sigmoid.((x .- expand_inst(m.μ, affine_shape)) ./ sqrt.(expand_inst(m.σ², affine_shape) .+ m.ϵ))), atol = 1.0e-7)
  end

  let m = InstanceNorm(2), sizes = (2, 4, 1, 2, 3),
      x = param(reshape(collect(1:prod(sizes)), sizes))
    y = reshape(permutedims(x, [3, 1, 2, 4, 5]), :, 2, 3)
    y = reshape(m(y), sizes...)
    @test m(x) == y
  end

  # check that μ, σ², and the output are the correct size for higher rank tensors
  let m = InstanceNorm(2), sizes = (5, 5, 3, 4, 2, 6),
      x = param(reshape(collect(1:prod(sizes)), sizes))
    y = m(x)
    @test size(m.μ) == (sizes[end - 1], )
    @test size(m.σ²) == (sizes[end - 1], )
    @test size(y) == sizes
  end

  # show that instance norm is equal to batch norm when channel and batch dims are squashed
  let m_inorm = InstanceNorm(2), m_bnorm = BatchNorm(12), sizes = (5, 5, 3, 4, 2, 6),
      x = param(reshape(collect(1:prod(sizes)), sizes))
    @test m_inorm(x) == reshape(m_bnorm(reshape(x, (sizes[1:end - 2]..., :, 1))), sizes)
  end

  let m = InstanceNorm(32), x = randn(Float32, 416, 416, 32, 1);
    m(x)
    @test (@allocated m(x)) <  100_000_000
  end

end
