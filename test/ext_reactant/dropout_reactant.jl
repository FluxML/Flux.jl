# Train-mode dropout under Reactant needs `NNlib.dropout` to accept a traced probability
# (https://github.com/FluxML/NNlib.jl/issues/807). Until an NNlib release has the fix, the tests that
# need it are marked broken.
function reactant_dropout_supported(dev)
    try
        Reactant.@jit Flux.NNlib.dropout(ones(Float32, 4) |> dev, 0.5)
        return true
    catch e
        e isa MethodError || rethrow()
        return false
    end
end

# Steps with a zero learning rate, returning the model's output at each step (the loss' aux output).
function dropout_outputs(model, x; nsteps = 3)
    loss(m, a) = (y = m(a); (sum(abs2, y), y))
    opt = Flux.setup(Descent(0f0), model)
    return [Flux.trainstep!(loss, model, (x,), opt)[2] for _ in 1:nsteps]
end

dropout_masks(model, x; kws...) = [iszero.(y) for y in dropout_outputs(model, x; kws...)]

@testset "Dropout on Reactant" begin
    dev = MLDataDevices.reactant_device(force=true)
    broken = !reactant_dropout_supported(dev)
    x = randn(Float32, 32, 64) |> dev

    @testset "fresh mask every step" begin
        model = trainmode!(Chain(Dense(32 => 32), Dropout(0.5)) |> dev)
        @test allunique(dropout_masks(model, x)) broken=broken
        @test mean(only(dropout_masks(model, x; nsteps=1))) ≈ 0.5 atol=0.05 broken=broken
    end

    @testset "dims" begin
        # `dims=1` draws one choice per row, shared across the batch.
        model = trainmode!(Chain(Dense(32 => 32), Dropout(0.5; dims=1)) |> dev)
        rowwise(mask) = all(allequal, eachrow(mask))
        @test all(rowwise, dropout_masks(model, x)) broken=broken
        @test allunique(dropout_masks(model, x)) broken=broken
    end

    # Doesn't call `NNlib.dropout`, so it works without the NNlib fix.
    @testset "AlphaDropout" begin
        p = 0.5f0
        model = trainmode!(Chain(Dense(32 => 32), AlphaDropout(p)) |> dev)
        # dropped entries are all set to `A * α′ + B`, see `AlphaDropout`
        α′ = -1.7580993408473766f0
        A = inv(sqrt((1 - p) * (1 + p * α′^2)))
        masks = [isapprox.(y, A * α′ * (1 - p); rtol=1e-4) for y in dropout_outputs(model, x)]
        @test allunique(masks)
        @test mean(first(masks)) ≈ p atol=0.05
    end

    @testset "independent masks in two layers" begin
        # Each layer gets its own RNG when moved to the device, unless they share one.
        for rng in (nothing, Xoshiro(1))
            drop() = rng === nothing ? Dropout(0.5) : Dropout(0.5; rng)
            model = trainmode!(Chain(Dense(32 => 32), Parallel(vcat, drop(), drop())) |> dev)
            agreement(mask) = mean(mask[1:32, :] .== mask[33:64, :])
            @test agreement(only(dropout_masks(model, x; nsteps=1))) ≈ 0.5 atol=0.05 broken=broken
        end
    end

    @testset "training with dropout reduces the loss" begin
        X = randn(Float32, 4, 64)
        Y = randn(Float32, 2, 4) * X
        model0 = Chain(Dense(4 => 32, relu), Dropout(0.2), Dense(32 => 2))
        model = trainmode!(deepcopy(model0) |> dev)
        opt = Flux.setup(Adam(1f-2), model)
        loss(m, a, b) = Flux.mse(m(a), b)
        function trained_loss()
            xy = (X, Y) |> dev
            for _ in 1:200
                Flux.trainstep!(loss, model, xy, opt)
            end
            return loss(testmode!(cpu(model)), X, Y)
        end
        @test trained_loss() < loss(testmode!(model0), X, Y) / 4 broken=broken
    end

    @testset "test mode is the identity" begin
        model = testmode!(Chain(Dense(32 => 32), Dropout(0.5)) |> dev)
        ys = dropout_outputs(model, x)
        @test allequal(ys)
        @test ys[1] ≈ Array(Reactant.@jit model[1](x))
    end
end
