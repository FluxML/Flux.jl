# These are used only in test_utils.jl but cannot leave there
# because Reactant is only optionally loaded and the macros fail when it is not loaded.

# On CUDA, XLA's default Float32 matmul and conv precision is TF32-like (relative error ~1e-4–1e-3
# against Float64), which `test_gradients`' tolerances can't absorb beyond tiny models. Compile at full
# precision so that the gradient tests check correctness rather than TF32 rounding.
reactant_full_precision(f) = Reactant.with_config(f;
    dot_general_precision = Reactant.PrecisionConfig.HIGHEST,
    convolution_precision = Reactant.PrecisionConfig.HIGHEST)

function reactant_withgradient(f, x...)
    y, g = reactant_full_precision() do
        Reactant.@jit Flux.withgradient(f, AutoEnzyme(), x...)
    end
    return y, g
end

function reactant_loss(loss, x...)
    l = reactant_full_precision(() -> Reactant.@jit loss(x...))
    @test l isa Reactant.ConcreteRNumber
    return l
end

# `test_gradients` on the Reactant device against Zygote, for each `(model, inputs, name, kwargs)` case.
function test_reactant_gradients(cases)
    for (model, xs, name, kws) in cases
        @testset "$name" begin
            @test test_gradients(model, xs...; reference = AutoZygote(), test_reactant = true, test_cpu = false, kws...)
        end
    end
end

reactant_on_cuda() = Reactant.XLA.platform_name(Reactant.XLA.default_backend()) == "cuda"

# Train the same MLP with `rule` eagerly and with `trainstep!` on the Reactant device. Returns the
# largest relative difference over the per-step losses and the final parameters.
function reactant_trajectory_error(rule; steps = 8)
    dev = MLDataDevices.reactant_device(force = true)
    model = Chain(Dense(4 => 8, tanh), Dense(8 => 2))
    X, Y = randn(Float32, 4, 16), randn(Float32, 2, 16)
    loss(m, x, y) = Flux.mse(m(x), y)

    eager, re = model, model |> dev
    x, y = (X, Y) |> dev
    opt_eager, opt_re = Flux.setup(rule, eager), Flux.setup(rule, re)
    err = 0.0
    for _ in 1:steps
        l_eager = Flux.trainstep!(loss, eager, (X, Y), opt_eager)
        l_re = Flux.trainstep!(loss, re, (x, y), opt_re)
        err = max(err, abs(l_re - l_eager) / l_eager)
    end
    flat(m) = reduce(vcat, vec.(Flux.trainables(m)))
    p_eager, p_re = flat(eager), flat(re |> cpu_device())
    return max(err, norm(p_re - p_eager) / norm(p_eager))
end

# `trainstep!` compiles at XLA's default precision, which is TF32-like for Float32 on CUDA.
reactant_trajectory_rtol() = reactant_on_cuda() ? 1e-3 : 1e-5
