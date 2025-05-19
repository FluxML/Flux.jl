evalwgrad(f, x...) = pullback(f, x...)[1]

@testset "Dropout" begin
  @testset for rng_kwargs in ((), (; rng = MersenneTwister()))
    x = [1.0+0im,2.0+1im,3.0+3im]
    @test x == Dropout(0.1; rng_kwargs...)(x)
    @test x == evalwgrad(Dropout(0; rng_kwargs...), x)
    @test zero(x) == evalwgrad(Dropout(1; rng_kwargs...), x)

    x = [1.,2.,3.]
    @test x == Dropout(0.1; rng_kwargs...)(x)
    @test x == evalwgrad(Dropout(0; rng_kwargs...), x)
    @test zero(x) == evalwgrad(Dropout(1; rng_kwargs...), x)

    x = rand(100)
    m = Dropout(0.9; rng_kwargs...)
    y = evalwgrad(m, x)
    @test count(a->a==0, y) > 50
    testmode!(m, true)
    y = evalwgrad(m, x) # should override istraining
    @test count(a->a==0, y) == 0
    testmode!(m, false)
    y = evalwgrad(m, x)
    @test count(a->a==0, y) > 50

    # Keyword active=false
    m2 = Dropout(0.9; active=false, rng_kwargs...)
    y2 = evalwgrad(m2, x)
    @test count(iszero, y2) == 0

    x = rand(Float32, 100)
    m = Chain(Dense(100 => 100),
              Dropout(0.9; rng_kwargs...))
    y = evalwgrad(m, x)
    @test count(a->a == 0, y) > 50
    testmode!(m, true)
    y = evalwgrad(m, x) # should override istraining
    @test count(a->a == 0, y) == 0

    x = rand(100, 50)
    m = Dropout(0.5; dims = 2, rng_kwargs...)
    y = m(x)
    c = map(i->count(a->a==0, @view y[i, :]), 1:100)
    @test minimum(c) == maximum(c)
    m = Dropout(0.5; dims = 1, rng_kwargs...)
    y = m(x)
    c = map(i->count(a->a==0, @view y[:, i]), 1:50)
    @test minimum(c) == maximum(c)

    # issue #1084
    m = Dropout(0.9; rng_kwargs...)
    x = rand(100)

    testmode!(m)
    y = m(x)
    @test count(a->a == 0, y) == 0
    trainmode!(m)
    y = m(x)
    @test count(a->a == 0, y) > 50

    y = Flux.dropout(values(rng_kwargs)..., x, 0.9) # , active=true)
    @test count(a->a == 0, y) > 50

    y = Flux.dropout(values(rng_kwargs)..., x, 0.9 * 0) # , active=false)
    @test count(a->a == 0, y) == 0

    # CPU RNGs map onto CPU ok
    if isempty(rng_kwargs)
      @test cpu(m).rng isa Random.TaskLocalRNG
    else
      @test cpu(m).rng === only(values(rng_kwargs))
    end
  end

  @test Dropout(0.5; active=true).active === true
  @test_throws Exception Dropout(0.5; active=:something_else)
end

@testset "AlphaDropout" begin
  @testset for rng_kwargs in ((), (; rng = MersenneTwister()))
    x = [1., 2., 3.]
    @test x == AlphaDropout(0.1; rng_kwargs...)(x)
    @test x == evalwgrad(AlphaDropout(0; rng_kwargs...), x)
    @test zero(x) == evalwgrad(AlphaDropout(1; rng_kwargs...), x)

    x = randn(1000) # large enough to prevent flaky test
    m = AlphaDropout(0.5; rng_kwargs...)
    q = 0.5
    u = mean(x)
    α′ = -1.7580993408473766

    y = evalwgrad(m, x)
    # Should preserve unit mean and variance
    @test mean(y) ≈ 0 atol=0.2
    @test var(y) ≈ 1 atol=0.2

    # Should check that the mean and variance matches the formula
    # E(xd + α′(1-d)) = qu + (1-q)α′
    @test mean(y) ≈ (q*u) + ((1-q)*α′)

    testmode!(m, true) # should override istraining
    @test evalwgrad(m, x) == x

    testmode!(m, false)
    y = evalwgrad(m, x)
    @test mean(y) ≈ 0 atol=0.2
    @test var(y) ≈ 1 atol=0.2

    # Known good value ranges
    # Values taken from https://github.com/pytorch/pytorch/blob/v1.10.0/test/cpp/api/modules.cpp#L1337-L1338
    x = ones(100)
    if isempty(rng_kwargs)
      @test 40 < sum(evalwgrad(m, x)) < 130
    else
      # FIXME: this breaks spuriously for MersenneTwister
      @test_skip 40 < sum(evalwgrad(m, x)) < 130
    end

    # CPU RNGs map onto CPU ok
    if isempty(rng_kwargs)
      @test cpu(m).rng isa Random.TaskLocalRNG
    else
      @test cpu(m).rng === only(values(rng_kwargs))
    end
  end

  @test AlphaDropout(0.5; active=true).active === true
  @test_throws Exception AlphaDropout(0.5; active=:something_else)
end

@testset "BatchNorm" begin
  let m = BatchNorm(2), x = [1.0 3.0 5.0;
                             2.0 4.0 6.0]

    @test Flux.hasaffine(m) == true
    @test length(Flux.trainables(m)) == 2

    @test m.β == [0, 0]  # initβ(2)
    @test m.γ == [1, 1]  # initγ(2)
    # initial m.σ is 1
    # initial m.μ is 0

    y = evalwgrad(m, x)
    @test isapprox(y, [-1.22474 0 1.22474; -1.22474 0 1.22474], atol = 1.0e-5)
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
    @test m.σ² ≈ .1 .* var(x, dims=2, corrected=false) .* (3 / 2).+ .9 .* [1., 1.]

    x′ = m(x)
    @test isapprox(x′[1], (1 .- 0.3) / sqrt(1.3), atol = 1.0e-5)

    @inferred m(x)
  end

  let m = BatchNorm(2; track_stats=false), x = Float32[1.0 3.0 5.0; 2.0 4.0 6.0]
    y = @inferred m(x)
    m16 = f16(m)
    y16 = @inferred m16(f16(x))
    @test eltype(y16) == Float16
    @test y16 ≈ y  atol=1e-3
  end

  # with activation function
  let m = BatchNorm(2, sigmoid), x = Float32[1.0 3.0 5.0;
                                             2.0 4.0 6.0]
    y = m(x)
    @test isapprox(y, sigmoid.((x .- m.μ) ./ sqrt.(m.σ² .+ m.ϵ)), atol = 1.0e-7)
    @inferred m(x)
    m16 = f16(m)
    y16 = @inferred m16(f16(x))
    @test eltype(y16) == Float16
    @test y16 ≈ y  atol=1e-3
  end

  let m = trainmode!(BatchNorm(2)), x = reshape(Float32.(1:6), 3, 2, 1)
    y = reshape(permutedims(x, [2, 1, 3]), 2, :)
    y = permutedims(reshape(m(y), 2, 3, 1), [2, 1, 3])
    @test m(x) == y
    @inferred m(x)
  end

  let m = trainmode!(BatchNorm(2)), x = reshape(Float32.(1:12), 2, 3, 2, 1)
    y = reshape(permutedims(x, [3, 1, 2, 4]), 2, :)
    y = permutedims(reshape(m(y), 2, 2, 3, 1), [2, 3, 1, 4])
    @test m(x) == y
    @inferred m(x)
  end

  let m = trainmode!(BatchNorm(2)), x = reshape(Float32.(1:24), 2, 2, 3, 2, 1)
    y = reshape(permutedims(x, [4, 1, 2, 3, 5]), 2, :)
    y = permutedims(reshape(m(y), 2, 2, 2, 3, 1), [2, 3, 4, 1, 5])
    @test m(x) == y
    @inferred m(x)
  end

  let m = BatchNorm(32), x = randn(Float32, 416, 416, 32, 1);
    m(x)
    @test (@allocated m(x)) <  100_000_000
    @inferred m(x)
  end

  @test length(Flux.trainables(BatchNorm(10))) == 2
  @test length(Flux.trainables(BatchNorm(10, affine=true))) == 2
  @test length(Flux.trainables(BatchNorm(10, affine=false))) == 0

  @test BatchNorm(5; active=true).active === true
  @test_throws Exception BatchNorm(5; active=:something_else)
end

@testset "InstanceNorm" begin
  # begin tests
  let m = InstanceNorm(2; affine=true, track_stats=true), sizes = (3, 2, 2),
        x = reshape(collect(1:prod(sizes)), sizes)

      @test length(Flux.trainables(m)) == 2
      x = Float32.(x)
      @test m.β == [0, 0]  # initβ(2)
      @test m.γ == [1, 1]  # initγ(2)
      y = evalwgrad(m, x)

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
      N = ndims(x)
      @test m.μ ≈ [0.5, 0.8]
      n = prod(size(x,i) for i in 1:N-2)
      corr = n / (n-1)
      σ² = var(x, dims=1:N-2, corrected=false)
      @test m.σ² ≈ 0.1*corr*vec(mean(σ², dims=N)) .+ 0.9 * 1

      y = m(x)
      @test length(m.μ) == 2
      @test length(m.σ²) == 2
      @test y ≈ (x .- reshape(m.μ, 1,2,1)) ./ sqrt.(reshape(m.σ², 1,2,1) .+ 1f-5)   atol=1.0e-5

      @inferred m(x)
  end

  # with activation function
  let m = InstanceNorm(2, sigmoid; affine=true, track_stats=true), sizes = (3, 2, 2),
      x = reshape(collect(1:prod(sizes)), sizes)
    x = Float64.(x)
    affine_shape = collect(sizes)
    affine_shape[[1,3]] .= 1

    y = evalwgrad(m, x)
    y = m(x) # inference time after a training step
    μ = reshape(m.μ, affine_shape...)
    σ² = reshape(m.σ², affine_shape...)
    @test y ≈ sigmoid.((x .- μ) ./ sqrt.(σ² .+ m.ϵ))   atol=1.0e-7

    @inferred m(x)
  end

  # with activation function
  let m = InstanceNorm(2, sigmoid; affine=true, track_stats=false), sizes = (3, 2, 2),
      x = reshape(collect(1:prod(sizes)), sizes)

    @test Flux.hasaffine(m) == true
    @test length(Flux.trainables(m)) == 2
    x = Float64.(x)
    y = m(x)
    μ = mean(x, dims=1)
    σ² = var(x, dims=1, corrected=false)
    @test y ≈ sigmoid.((x .- μ) ./ sqrt.(σ² .+ m.ϵ))   atol=1.0e-7

    @inferred m(x)
  end

  let m = InstanceNorm(2, sigmoid), sizes = (3, 2, 2),
      x = reshape(collect(1:prod(sizes)), sizes)
    @test Flux.hasaffine(m) == false
    @test length(Flux.trainables(m)) == 0

    x = Float64.(x)
    y = m(x)
    μ = mean(x, dims=1)
    σ² = var(x, dims=1, corrected=false)
    @test y ≈ sigmoid.((x .- μ) ./ sqrt.(σ² .+ m.ϵ))   atol=1.0e-7

    @inferred m(x)
  end


  let m = trainmode!(InstanceNorm(2; affine=true)), sizes = (2, 4, 1, 2, 3),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    y = reshape(permutedims(x, [3, 1, 2, 4, 5]), :, 2, 3)
    y = reshape(m(y), sizes...)
    @test m(x) == y

    @inferred m(x)
  end

  # check that μ, σ², and the output are the correct size for higher rank tensors
  let m = InstanceNorm(2; affine=true,track_stats=true), sizes = (5, 5, 3, 4, 2, 6),
      x = reshape(Float32.(collect(1:prod(sizes))), sizes)
    y = evalwgrad(m, x)
    @test size(m.μ) == (sizes[end - 1], )
    @test size(m.σ²) == (sizes[end - 1], )
    @test size(y) == sizes

    @inferred m(x)
  end

  # show that instance norm is equal to batch norm when channel and batch dims are squashed
  let m_inorm = trainmode!(InstanceNorm(2; affine=true)), m_bnorm = trainmode!(BatchNorm(12)), sizes = (5, 5, 3, 4, 2, 6),
      x = reshape(Float32.(collect(1:prod(sizes))), sizes)
    @test m_inorm(x) == reshape(m_bnorm(reshape(x, (sizes[1:end - 2]..., :, 1))), sizes)
  end

  let m = InstanceNorm(32), x = randn(Float32, 416, 416, 32, 1);
    m(x)
    @test (@allocated m(x)) <  100_000_000

    @inferred m(x)
  end

  @test length(Flux.trainables(InstanceNorm(10))) == 0
  @test length(Flux.trainables(InstanceNorm(10, affine=true))) == 2
  @test length(Flux.trainables(InstanceNorm(10, affine=false))) == 0

  @test InstanceNorm(5; active=true).active === true
  @test_throws Exception InstanceNorm(5; active=:something_else)
end

@testset "LayerNorm" begin
  x = rand(2,3)
  @test LayerNorm(2)(x) ≈ Flux.normalise(x, dims=1)
  x = rand(2,3,4)
  @test LayerNorm(2)(x) ≈ Flux.normalise(x, dims=1)
  x = rand(2,3,4,5)
  @test LayerNorm(2)(x) ≈ Flux.normalise(x, dims=1)
  x = rand(2)
  @test LayerNorm(2, tanh)(x) ≈ tanh.(Flux.normalise(x, dims=1))

  x = rand(2,3,4,5)
  @test LayerNorm((2,3))(x) ≈ Flux.normalise(x, dims=(1,2))
  x = rand(2,3,4,5)
  @test LayerNorm((2,3,4))(x) ≈ Flux.normalise(x, dims=1:3)

  m = LayerNorm((2,3,4))
  @test Flux.hasaffine(m) == true
  @test length(Flux.trainables(m)) == 2
  m = LayerNorm((2,3,4), affine=false)
  @test Flux.hasaffine(m) == false
  @test length(Flux.trainables(m)) == 0
end

@testset "GroupNorm" begin
  # begin tests
  squeeze(x) = dropdims(x, dims = tuple(findall(size(x) .== 1)...)) # To remove all singular dimensions

  let m = GroupNorm(4,2), sizes = (3,4,2),
        x = reshape(collect(1:prod(sizes)), sizes)

      @test length(Flux.trainables(m)) == 2
      x = Float32.(x)
      @test m.β == [0, 0, 0, 0]  # initβ(32)
      @test m.γ == [1, 1, 1, 1]  # initγ(32)

      ŷ = evalwgrad(m, x)
    
      @test m.μ === nothing
      @test m.σ² === nothing
      ŷ = m(x)
      y = [-1.4638476 0.29276943 -1.4638476 0.29276943; -0.87830865 0.87830853 -0.8783088 0.8783083; -0.29276967 1.4638474 -0.2927699 1.4638472;;; -1.4638476 0.29276943 -1.4638472 0.29276943; -0.8783083 0.8783083 -0.8783083 0.8783083; -0.29276943 1.4638472 -0.29276943 1.4638472]

      @test ŷ ≈ y   atol=1.0e-5
  end
  # with activation function
  let m = GroupNorm(4,2, sigmoid), sizes = (3, 4, 2),
      x = reshape(collect(1:prod(sizes)), sizes)
    
    x = Float32.(x)
    μ_affine_shape = ones(Int,length(sizes) + 1)
    μ_affine_shape[end-1] = 2 # Number of groups

    affine_shape = ones(Int,length(sizes) + 1)
    affine_shape[end-2] = 2 # Channels per group
    affine_shape[end-1] = 2 # Number of groups
    affine_shape[1] = sizes[1]
    affine_shape[end] = sizes[end]

    og_shape = size(x)

    ŷ = m(x)
    y = [0.18787955 0.57267404 0.18787955 0.57267404; 0.2935284 0.70647156 0.29352835 0.70647156; 0.42732593 0.81212044 0.42732587 0.8121204;;; 0.18787955 0.57267404 0.1878796 0.57267404; 0.29352847 0.70647156 0.29352847 0.70647156; 0.42732602 0.8121204 0.42732602 0.8121204]
    @test ŷ ≈ y   atol=1e-7
  end

  let m = trainmode!(GroupNorm(2,2)), sizes = (2, 4, 1, 2, 3),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    y = reshape(permutedims(x, [3, 1, 2, 4, 5]), :, 2, 3)
    y = reshape(m(y), sizes...)
    @test m(x) == y
  end

  # show that group norm is the same as instance norm when the group size is the same as the number of channels
  let IN = trainmode!(InstanceNorm(4; affine=true)), GN = trainmode!(GroupNorm(4,4)), sizes = (2,2,3,4,5),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    @test IN(x) ≈ GN(x)
  end

  # show that group norm is the same as batch norm for a group of size 1 and batch of size 1
  let BN = trainmode!(BatchNorm(4)), GN = trainmode!(GroupNorm(4,4)), sizes = (2,2,3,4,1),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    @test BN(x) ≈ GN(x)
  end

  @test GroupNorm(5, 5; active=true).active === true
  @test_throws Exception GroupNorm(5, 5; active=:something_else)
end

@testset "second derivatives" begin
  m1 = Dropout(0.5)
  @test Zygote.hessian_reverse(sum∘m1, [1.0,2.0,3.0]) == zeros(3, 3)

  m2 = Chain(BatchNorm(3), sum)
  @test Zygote.hessian_reverse(m2, Float32[1 2; 3 4; 5 6]) == zeros(Float32, 6, 6) broken = VERSION >= v"1.11"
end

@testset "ForwardDiff" begin
  bn = BatchNorm(3)
  @test ForwardDiff.jacobian(bn, rand(Float32, 3, 4)) isa Matrix{Float32}
  # iszero(bn.μ)  # is true. But ideally would not be, if Flux would automatically choose trainmode
  Flux.trainmode!(bn)
  # This was an error, https://github.com/FluxML/Flux.jl/issues/2122
  @test ForwardDiff.jacobian(bn, rand(Float32, 3, 4)) isa Matrix{Float32}
  @test !iszero(bn.μ)

  # Easy case of 2122, gradient with x
  x5 = rand(Float32, 5, 3)
  bn1 = BatchNorm(5, relu)
  bn2 = BatchNorm(5, relu)
  g1 = Zygote.gradient(x -> sum(abs2, bn1(x)), x5)[1]
  g2 = ForwardDiff.gradient(x -> sum(abs2, bn2(x)), x5)
  @test g1 ≈ g2

  # Harder case? 
  v1, re1 = Flux.destructure(BatchNorm(5, relu));
  g1 = Zygote.gradient(v -> sum(abs2, re1(v)(x5)), v1)[1]

  v2, re2 = Flux.destructure(BatchNorm(5, relu));
  g2 = ForwardDiff.gradient(v -> sum(abs2, re2(v)(x5)), v2)
end

