using Flux, Test, Statistics
using Zygote: pullback, ForwardDiff

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

    x = rand(Float32, 100)
    m = Chain(Dense(100,100),
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

    y = Flux.dropout(values(rng_kwargs)..., x, 0.9, active=true)
    @test count(a->a == 0, y) > 50

    y = Flux.dropout(values(rng_kwargs)..., x, 0.9, active=false)
    @test count(a->a == 0, y) == 0

    # CPU RNGs map onto CPU ok
    if isempty(rng_kwargs)
      if VERSION >= v"1.7"
        @test cpu(m).rng isa Random.TaskLocalRNG
      else
        @test cpu(m).rng isa Random._GLOBAL_RNG
      end
    else
      @test cpu(m).rng === only(values(rng_kwargs))
    end
  end

  for active in (true, false)
    m = Dropout(0.5, :, active)
    _, back = @inferred pullback(m, rand(10)) # _, DropoutPullback{Array{Float64}}
    @inferred back(ones(10)) # Array{Float64}
  end
end

@testset "AlphaDropout" begin
  @testset for rng_kwargs in ((), (; rng = MersenneTwister()))
    x = [1., 2., 3.]
    @test x == AlphaDropout(0.1; rng_kwargs...)(x)
    @test x == evalwgrad(AlphaDropout(0; rng_kwargs...), x)
    @test zero(x) == evalwgrad(AlphaDropout(1; rng_kwargs...), x)

    x = randn(1000) # large enough to prevent flaky test
    m = AlphaDropout(0.5; rng_kwargs...)

    y = evalwgrad(m, x)
    # Should preserve unit mean and variance
    @test mean(y) ≈ 0 atol=0.2
    @test var(y) ≈ 1 atol=0.2

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
      if VERSION >= v"1.7"
        @test cpu(m).rng isa Random.TaskLocalRNG
      else
        @test cpu(m).rng isa Random._GLOBAL_RNG
      end
    else
      @test cpu(m).rng === only(values(rng_kwargs))
    end
  end
end

@testset "BatchNorm" begin
  let m = BatchNorm(2), x = [1.0 3.0 5.0;
                             2.0 4.0 6.0]

    @test Flux.hasaffine(m) == true
    @test length(Flux.params(m)) == 2

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

  let m = BatchNorm(2; track_stats=false), x = [1.0 3.0 5.0; 2.0 4.0 6.0]
    @inferred m(x)
  end

  # with activation function
  let m = BatchNorm(2, sigmoid), x = [1.0 3.0 5.0;
                                      2.0 4.0 6.0]
    y = m(x)
    @test isapprox(y, sigmoid.((x .- m.μ) ./ sqrt.(m.σ² .+ m.ϵ)), atol = 1.0e-7)
    @inferred m(x)
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

  @test length(Flux.params(BatchNorm(10))) == 2
  @test length(Flux.params(BatchNorm(10, affine=true))) == 2
  @test length(Flux.params(BatchNorm(10, affine=false))) == 0
end

@testset "InstanceNorm" begin
  # begin tests
  let m = InstanceNorm(2; affine=true, track_stats=true), sizes = (3, 2, 2),
        x = reshape(collect(1:prod(sizes)), sizes)

      @test length(Flux.params(m)) == 2
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
    @test length(Flux.params(m)) == 2
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
    @test length(Flux.params(m)) == 0

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

  @test length(Flux.params(InstanceNorm(10))) == 0
  @test length(Flux.params(InstanceNorm(10, affine=true))) == 2
  @test length(Flux.params(InstanceNorm(10, affine=false))) == 0
end

@testset "LayerNorm" begin
  x = rand(2,3)
  @test LayerNorm(2)(x) ≈ Flux.normalise(x, dims=1)
  x = rand(2,3,4)
  @test LayerNorm(2)(x) ≈ Flux.normalise(x, dims=1)
  x = rand(2,3,4,5)
  @test LayerNorm(2)(x) ≈ Flux.normalise(x, dims=1)

  x = rand(2)
  m = LayerNorm(2, tanh)
  @test m(x) ≈ tanh.(Flux.normalise(x, dims=1))
  _, back = @inferred pullback(|>, x, m)
  # TODO needs https://github.com/FluxML/Zygote.jl/pull/1248
  # @inferred back(1.0)

  x = rand(2,3,4,5)
  @test LayerNorm((2,3))(x) ≈ Flux.normalise(x, dims=(1,2))
  x = rand(2,3,4,5)
  @test LayerNorm((2,3,4))(x) ≈ Flux.normalise(x, dims=1:3)

  m = LayerNorm((2,3,4))
  @test Flux.hasaffine(m) == true
  @test length(Flux.params(m)) == 2
  m = LayerNorm((2,3,4), affine=false)
  @test Flux.hasaffine(m) == false
  @test length(Flux.params(m)) == 0
end

@testset "GroupNorm" begin
  # begin tests
  squeeze(x) = dropdims(x, dims = tuple(findall(size(x) .== 1)...)) # To remove all singular dimensions

  let m = GroupNorm(4,2, track_stats=true), sizes = (3,4,2),
        x = reshape(collect(1:prod(sizes)), sizes)

      @test length(Flux.params(m)) == 2
      x = Float32.(x)
      @test m.β == [0, 0, 0, 0]  # initβ(32)
      @test m.γ == [1, 1, 1, 1]  # initγ(32)

      y = evalwgrad(m, x)

      #julia> x
      #[:, :, 1]  =
      # 1.0  4.0  7.0  10.0
      # 2.0  5.0  8.0  11.0
      # 3.0  6.0  9.0  12.0
      #
      #[:, :, 2] =
      # 13.0  16.0  19.0  22.0
      # 14.0  17.0  20.0  23.0
      # 15.0  18.0  21.0  24.0
      #
      # μ will be
      # (1. + 2. + 3. + 4. + 5. + 6.) / 6 = 3.5
      # (7. + 8. + 9. + 10. + 11. + 12.) / 6 = 9.5
      #
      # (13. + 14. + 15. + 16. + 17. + 18.) / 6 = 15.5
      # (19. + 20. + 21. + 22. + 23. + 24.) / 6 = 21.5
      #
      # μ =
      # 3.5   15.5
      # 9.5   21.5
      #
      # ∴ update rule with momentum:
      # (1. - .1) * 0 + .1 * (3.5 + 15.5) / 2 = 0.95
      # (1. - .1) * 0 + .1 * (9.5 + 21.5) / 2 = 1.55
      @test m.μ ≈ [0.95, 1.55]
      n = prod(size(x)) ÷ m.G ÷ size(x)[end]
      corr = n / (n-1)
      z = reshape(x,3,2,2,2)
      σ² = var(z, dims=(1,2), corrected=false)
      @test m.σ² ≈ 0.1*corr*vec(mean(σ², dims=4)) .+ 0.9 * 1

      y = m(x)
      out = (z .- reshape(m.μ, 1,1,2,1)) ./ sqrt.(reshape(m.σ², 1,1,2,1) .+ 1f-5)
      @test y ≈ reshape(out, size(x))   atol=1.0e-5
  end
  # with activation function
  let m = GroupNorm(4,2, sigmoid, track_stats=true), sizes = (3, 4, 2),
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

    y = m(x)
    x_ = reshape(x,affine_shape...)
    out = reshape(sigmoid.((x_ .- reshape(m.μ,μ_affine_shape...)) ./ sqrt.(reshape(m.σ²,μ_affine_shape...) .+ m.ϵ)),og_shape)
    @test y ≈ out   atol=1e-7
  end

  let m = trainmode!(GroupNorm(2,2, track_stats=true)), sizes = (2, 4, 1, 2, 3),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    y = reshape(permutedims(x, [3, 1, 2, 4, 5]), :, 2, 3)
    y = reshape(m(y), sizes...)
    @test m(x) == y
  end

  # check that μ, σ², and the output are the correct size for higher rank tensors
  let m = GroupNorm(4,2, track_stats=true), sizes = (5, 5, 3, 4, 4, 6),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    y = evalwgrad(m, x)
    @test size(m.μ) == (m.G,)
    @test size(m.σ²) == (m.G,)
    @test size(y) == sizes
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
end

@testset "second derivatives" begin
  m1 = Dropout(0.5)
  @test Zygote.hessian_reverse(sum∘m1, [1.0,2.0,3.0]) == zeros(3, 3)

  m2 = Chain(BatchNorm(3), sum)
  @test Zygote.hessian_reverse(m2, Float32[1 2; 3 4; 5 6]) == zeros(Float32, 6, 6)
end

@testset "ForwardDiff" begin
  bn = BatchNorm(3)
  @test ForwardDiff.jacobian(bn, rand(Float32, 3, 4)) isa Matrix{Float32}
  # iszero(bn.μ)  # is true. But ideally would not be, if Flux would automatically choose trainmode
  Flux.trainmode!(bn)
  # This was an error, https://github.com/FluxML/Flux.jl/issues/2122
  @test ForwardDiff.jacobian(bn, rand(Float32, 3, 4)) isa Matrix{Float32}
  @test !iszero(bn.μ)
end

