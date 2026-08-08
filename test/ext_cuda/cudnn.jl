
@testset "CUDNN BatchNorm" begin
    # TODO(#2700): NNlib 0.9.41 requires Float32 scale/bias for Float16 input; `f16`
    # normalization params are being reworked in Flux.jl#2700. Re-enable (Float16, f16) then.
    @testset "4D Input, $T" for (T,f) in [(Float32, identity)] # , (Float16, f16)
        x = randn(T, 2, 2, 3, 4)
        m = f(BatchNorm(3))
        gx = gpu(x)
        gm = gpu(m)

        y, back = pullback((m, x) -> m(x), m, x)
        gy, gback = pullback((m, x) -> m(x), gm, gx)

        @test cpu(gy) ≈ y  rtol=1e-3
        @test eltype(gy) == T
        @test eltype(gm(gx)) == T

        Δ = randn(T, size(y))
        dm, dx = back(Δ)
        gdm, gdx = gback(f(gpu(Δ)))

        @test dm.γ ≈ cpu(gdm.γ)
        @test dm.β ≈ cpu(gdm.β)
        @test dx ≈ cpu(gdx)
        @test eltype(gdm.γ) == T
        @test eltype(gdx) == T
    end

    @testset "2D Input" begin
        x = rand(Float32, 3, 4)
        m = BatchNorm(3)
        gx = gpu(x)
        gm = gpu(m)

        y, back = pullback((m, x) -> m(x), m, x)
        gy, gback = pullback((m, x) -> m(x), gm, gx)

        @test cpu(gy) ≈ y

        Δ = randn(Float32, size(y))
        dm, dx = back(Δ)
        gdm, gdx = gback(gpu(Δ))

        @test dm.γ ≈ cpu(gdm.γ)
        @test dm.β ≈ cpu(gdm.β)
        @test dx ≈ cpu(gdx)
    end
end
