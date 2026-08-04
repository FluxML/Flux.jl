@testset "autocast on GPU ($T)" for T in (Float16, BFloat16)
    x2 = CUDA.randn(Float32, 3, 8)
    x4 = CUDA.randn(Float32, 8, 8, 2, 3)

    @testset "eltype flow" begin
        model = Chain(Dense(3 => 4, relu), BatchNorm(4), Dense(4 => 2)) |> gpu
        y = autocast(() -> model(x2), T)
        @test y isa CuArray{T}

        c = Chain(Conv((3, 3), 2 => 4, relu), BatchNorm(4), MaxPool((2, 2))) |> gpu
        yc = autocast(() -> c(x4), T)
        @test eltype(yc) == T
    end

    @testset "training step" begin
        model = Chain(Dense(3 => 4, relu), BatchNorm(4), Dense(4 => 2)) |> gpu
        ytarget = CUDA.randn(Float32, 2, 8)
        loss(m) = Flux.mse(m(x2), ytarget)

        val, grad = Flux.withgradient(loss, model; autocast=T)
        @test val isa Float32
        gW = grad[1].layers[1].weight
        @test gW isa CuArray{Float32}
        @test all(isfinite, Array(gW))

        # close to the full-precision gradient
        val32, grad32 = Flux.withgradient(loss, model)
        rtol = T == Float16 ? 0.03 : 0.15
        @test val ≈ val32 rtol=rtol
        @test Array(gW) ≈ Array(grad32[1].layers[1].weight) rtol=rtol atol=0.05

        # parameters stay Float32 through a train! step
        opt = Flux.setup(Adam(1e-3), model)
        data = [(CUDA.randn(Float32, 3, 8), CUDA.randn(Float32, 2, 8)) for _ in 1:3]
        Flux.train!((m, x, y) -> Flux.mse(m(x), y), model, data, opt; autocast=T)
        @test model[1].weight isa CuArray{Float32}
    end

    @testset "conv gradient" begin
        c = Conv((3, 3), 2 => 4) |> gpu
        g = Flux.gradient(m -> sum(abs2, m(x4)), c; autocast=T)[1]
        @test g.weight isa CuArray{Float32}
        g32 = Flux.gradient(m -> sum(abs2, m(x4)), c)[1]
        @test Array(g.weight) ≈ Array(g32.weight) rtol=0.15 atol=0.1
    end
end
