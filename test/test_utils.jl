# group here all losses, used in tests
const ALL_LOSSES = [Flux.Losses.mse, Flux.Losses.mae, Flux.Losses.msle,
                    Flux.Losses.crossentropy, Flux.Losses.logitcrossentropy,
                    Flux.Losses.binarycrossentropy, Flux.Losses.logitbinarycrossentropy,
                    Flux.Losses.kldivergence,
                    Flux.Losses.huber_loss,
                    Flux.Losses.tversky_loss,
                    Flux.Losses.dice_coeff_loss,
                    Flux.Losses.poisson_loss,
                    Flux.Losses.hinge_loss, Flux.Losses.squared_hinge_loss,
                    Flux.Losses.binary_focal_loss, Flux.Losses.focal_loss,
                    Flux.Losses.siamese_contrastive_loss]


const TEST_MODELS = [
    (Dense(2=>4), randn(Float32, 2), "Dense"),
    (Chain(Dense(2=>4, tanh), Dense(4=>3)), randn(Float32, 2), "Chain(Dense, Dense)"),
    (f64(Chain(Dense(2=>4), Dense(4=>2))), randn(Float64, 2, 1), "f64(Chain(Dense, Dense))"),
    (Flux.Scale([1.0f0 2.0f0 3.0f0 4.0f0], true, abs2), randn(Float32, 2), "Flux.Scale"),
    (Conv((3, 3), 2 => 3), randn(Float32, 3, 3, 2, 1), "Conv"),
    (Chain(Conv((3, 3), 2 => 3, ), Conv((3, 3), 3 => 1, tanh)), rand(Float32, 5, 5, 2, 1), "Chain(Conv, Conv)"),
    (Chain(Conv((4, 4), 2 => 2, pad=SamePad()), MeanPool((5, 5), pad=SamePad())), rand(Float32, 5, 5, 2, 2), "Chain(Conv, MeanPool)"),
    (Maxout(() -> Dense(5 => 4, tanh), 3), randn(Float32, 5, 1), "Maxout"),
    (SkipConnection(Dense(2 => 2), vcat), randn(Float32, 2, 3), "SkipConnection"),
    (Flux.Bilinear((2, 2) => 3), randn(Float32, 2, 1), "Bilinear"),
    (ConvTranspose((3, 3), 3 => 2, stride=2), rand(Float32, 5, 5, 3, 1), "ConvTranspose"),
    (LayerNorm(2), randn(Float32, 2, 10), "LayerNorm"),
    (BatchNorm(2), randn(Float32, 2, 10), "BatchNorm"),
    (first ∘ MultiHeadAttention(16), randn32(16, 20, 2), "MultiHeadAttention"),
    (RNN(3 => 2), randn(Float32, 3, 2), "RNN"), 
    (LSTM(3 => 5), randn(Float32, 3, 2), "LSTM"),
    (GRU(3 => 5), randn(Float32, 3, 10), "GRU"),
    (Chain(RNN(3 => 4), RNN(4 => 3)), randn(Float32, 3, 2), "Chain(RNN, RNN)"),
    (Chain(LSTM(3 => 5), LSTM(5 => 3)), randn(Float32, 3, 2), "Chain(LSTM, LSTM)"),
]

function _contains_no_numerical(kp, x)
    count = 0
    fmap(x) do y
        if y isa AbstractArray{<:AbstractFloat}
            count += 1
        end
        return y
    end
    return count == 0
end

function check_equal_leaves(a, b; rtol=1e-4, atol=1e-4)
    # Since Zygote could use nothing for an entire subtree, we prune the
    # the tree using _contains_no_numerical
    fmapstructure_with_path(a, b, exclude=_contains_no_numerical) do kp, x, y
        # @show kp
        if x isa AbstractArray{<:AbstractFloat}
            @test x ≈ y rtol=rtol atol=atol
        end
    end
    return true
end

function _contains_no_numerical(kp, x)
    count = 0
    fmap(x) do y
        if y isa AbstractArray{<:AbstractFloat}
            count += 1
        end
        return y
    end
    return count == 0
end

_default_fdm() = FiniteDifferences.central_fdm(5, 1, max_range=1e-2)

"""
Compare the `reference` and `compare` AD backends on the gradients of `f` at `xs...`.
The loss function can be customized (default is mean over outputs).

- If `test_gpu` is true, the `compare` backend is tested on GPU.
- If `test_cpu` is true, the `compare` backend is tested on CPU.
- If `test_reactant` is true, the Enzyme backend is tested with Reactant.
  Depending on the platform, this may run on CPU or GPU.
"""
function test_gradients(
            f, 
            xs...;
            rtol=1e-4, atol=1e-4,
            test_gpu = false,
            test_cpu = true,
            test_reactant = false,
            reference = AutoFiniteDifferences(; fdm = _default_fdm()),
            compare = AutoZygote(),
            loss = (f, xs...) -> mean(f(xs...)),
            test_mode = false,
            )

    @assert reference !== nothing "reference AD backend must be provided"
    @assert compare !== nothing || test_gpu "compare AD backend must be provided if test_gpu=false"
    compare = compare === nothing ? reference : compare

    if test_mode
        Flux.testmode!(f)
    else
        Flux.trainmode!(f)
    end

    cpu_dev = cpu_device()
    
    if test_gpu
        gpu_dev = gpu_device(force=true)
        cpu_dev = cpu_device()
        xs_gpu = xs |> gpu_dev
        f_gpu = f |> gpu_dev
    end
    
    if test_reactant
        reactant_dev = MLDataDevices.reactant_device(force=true)
        xs_re = xs |> reactant_dev
        f_re = f |> reactant_dev
    end

    ## Let's make sure first that the forward pass works.
    l = loss(f, xs...)
    @test l isa Number

    # Compute reference gradients in f64 precision
    y, gs = Flux.withgradient(loss, reference, Flux.f64(f), Flux.f64(xs)...)
    @test l ≈ y rtol=rtol atol=atol

    if test_cpu
        y2, gs2 = Flux.withgradient(loss, compare, f, xs...)
        @test l ≈ y2 rtol=rtol atol=atol
        check_equal_leaves(gs, gs2; rtol, atol)
    end

    if test_gpu
        l_gpu = loss(f_gpu, xs_gpu...)
        @test l_gpu isa Number

        y_gpu, gs_gpu = Flux.withgradient(loss, compare, f_gpu, xs_gpu...)
        @test l_gpu ≈ y_gpu rtol=rtol atol=atol
        check_equal_leaves(gs, gs_gpu |> cpu_dev; rtol, atol)  
    end

    if test_reactant
        l_re = reactant_loss(loss, f_re, xs_re...)
        @test l ≈ l_re rtol=rtol atol=atol

        y_re, g_re = reactant_withgradient(loss, f_re, xs_re...)
        @test y ≈ y_re rtol=rtol atol=atol
        check_equal_leaves(gs, g_re |> cpu_dev; rtol, atol)
    end

    return true
end
