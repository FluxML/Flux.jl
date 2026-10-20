# Gradient parity against Zygote for every loss, through a small model. Only the gradients w.r.t. the
# model and its input are compared, the labels are passed as `Const`: gradients w.r.t. the labels may
# legitimately differ between backends (e.g. `binarycrossentropy` at 0/1 labels, from the `xlogy` rules).
function test_loss_gradients(lossfn, model, x, y; rtol = 1e-4, atol = 1e-4)
    dev = MLDataDevices.reactant_device(force = true)
    loss(m, x, y) = lossfn(m(x), y)

    l = loss(model, x, y)
    l64, g64 = Flux.withgradient((m, x) -> loss(m, x, y), AutoZygote(), Flux.f64(model), Flux.f64(x))
    @test l ≈ l64 rtol=rtol atol=atol

    m_re, x_re, y_re = (model, x, y) |> dev
    l_re, g_re = reactant_withgradient(loss, m_re, x_re, Const(y_re))
    @test l_re ≈ l64 rtol=rtol atol=atol
    @test g_re[3] === nothing
    check_equal_leaves(g64, g_re[1:2] |> cpu_device(); rtol, atol)
    return true
end

@testset "Reactant loss gradients" begin
    r(dims...) = randn(Float32, dims...)
    x = r(4, 6)
    logits = Dense(4 => 3)
    probs = Chain(Dense(4 => 3), softmax)
    binprobs = Chain(Dense(4 => 3), sigmoid)
    positive = Chain(Dense(4 => 3), softplus)

    yreal = r(3, 6)
    yprob = softmax(r(3, 6))
    ybin = Float32.(rand(Bool, 3, 6))
    ypos = abs.(r(3, 6)) .+ 0.1f0
    ysign = 2 .* ybin .- 1
    yhot = onehotbatch(rand(1:3, 6), 1:3)

    L = Flux.Losses
    # loss => (model, labels)
    cases = Dict(
        L.mse => (logits, yreal), L.mae => (logits, yreal), L.huber_loss => (logits, yreal),
        L.msle => (positive, ypos), L.poisson_loss => (positive, ypos),
        L.crossentropy => (probs, yprob), L.logitcrossentropy => (logits, yprob),
        L.kldivergence => (probs, yprob), L.focal_loss => (probs, yprob),
        L.binarycrossentropy => (binprobs, ybin), L.logitbinarycrossentropy => (logits, ybin),
        L.binary_focal_loss => (binprobs, ybin),
        L.tversky_loss => (binprobs, ybin), L.dice_coeff_loss => (binprobs, ybin),
        L.hinge_loss => (logits, ysign), L.squared_hinge_loss => (logits, ysign),
        L.siamese_contrastive_loss => (logits, ybin),
    )
    @test issetequal(keys(cases), ALL_LOSSES)

    for lossfn in ALL_LOSSES
        @testset "$lossfn" begin
            model, y = cases[lossfn]
            @test test_loss_gradients(lossfn, model, x, y)
        end
    end

    @testset "logitcrossentropy, onehot labels" begin
        @test test_loss_gradients(L.logitcrossentropy, logits, x, yhot)
    end
    @testset "$lossfn, integer labels" for (lossfn, model) in ((L.logitcrossentropy, logits), (L.crossentropy, probs))
        @test test_loss_gradients(lossfn, model, x, rand(1:3, 6))
        @test test_loss_gradients((ŷ, y) -> lossfn(ŷ, y; agg = sum), model, x, rand(1:3, 6))
        @test test_loss_gradients((ŷ, y) -> lossfn(permutedims(ŷ), y; dims = 2), model, x, rand(1:3, 6))
    end
    @testset "onehot labels are gathered, not materialized" begin
        ŷ_re, yhot_re = (r(3, 6), yhot) |> MLDataDevices.reactant_device(force = true)
        @test !occursin("scatter", repr(Reactant.@code_hlo L.logitcrossentropy(ŷ_re, yhot_re)))
    end
    @testset "crossentropy with label_smoothing, agg=sum" begin
        @test test_loss_gradients((ŷ, y) -> L.crossentropy(ŷ, Flux.label_smoothing(y, 0.1f0); agg = sum), probs, x, Float32.(yhot))
    end
    @testset "BFloat16 is computed in Float32: $name" for (name, lossfn, y) in (
            ("logitcrossentropy, labels", L.logitcrossentropy, rand(1:3, 6)),
            ("logitcrossentropy, onehot", L.logitcrossentropy, yhot),
            ("mse", L.mse, bf16(yreal)),
            ("crossentropy", (ŷ, y) -> L.crossentropy(softmax(ŷ), y), bf16(yprob)))
        loss(m, x, y) = lossfn(m(x), y)
        model, xb = bf16(logits), bf16(x)
        l = loss(model, xb, y)
        m_re, x_re, y_re = (model, xb, y) |> MLDataDevices.reactant_device(force = true)
        l_re, g_re = reactant_withgradient(loss, m_re, x_re, Const(y_re))
        @test l_re isa Reactant.ConcreteRNumber{Float32}
        @test l_re ≈ l rtol=1e-2
        @test eltype(g_re[1].weight) == BFloat16
    end
end
