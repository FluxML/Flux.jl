using Flux, CuArrays, Test
using Flux: forward

@testset "CUDNN BatchNorm" begin
    @testset "4D Input" begin
        x = Float64.(collect(reshape(1:12, 2, 2, 3, 1)))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y, back = forward((m, x) -> m(x), m, x)
        cy, cback = forward((m, x) -> m(x), cm, cx)

        @test cpu(cy) ≈ y

        Δ = randn(size(y))
        dm, dx = back(Δ)
        cdm, cdx = cback(gpu(Δ))

        @test dm[].γ ≈ cpu(cdm[].γ)
        @test dm[].β ≈ cpu(cdm[].β)
        @test dx ≈ cpu(cdx)
    end

    @testset "2D Input" begin
        x = Float64.(collect(reshape(1:12, 3, 4)))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y, back = forward((m, x) -> m(x), m, x)
        cy, cback = forward((m, x) -> m(x), cm, cx)

        @test cpu(cy) ≈ y

        Δ = randn(size(y))
        dm, dx = back(Δ)
        cdm, cdx = cback(gpu(Δ))

        @test dm[].γ ≈ cpu(cdm[].γ)
        @test dm[].β ≈ cpu(cdm[].β)
        @test dx ≈ cpu(cdx)
    end
end
