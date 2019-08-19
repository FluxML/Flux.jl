using Flux, CuArrays, Test
trainmode(f, x...) = forward(f, x...)[1]

@testset "CUDNN BatchNorm" begin
    @testset "4D Input" begin
        x = Float64.(collect(reshape(1:12, 2, 2, 3, 1)))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y = trainmode(m, x)
        cy = trainmode(cm, cx)

        @test cpu(cy) ≈ y

        g = gradient(()->sum(m(x)), params(m))
        cg = gradient(()->sum(cm(cx)), params(cm))

        @test g[m.γ] ≈ cpu(cg[cm.γ])
        @test g[m.β] ≈ cpu(cg[cm.β])
    end

    @testset "2D Input" begin
        x = Float64.(collect(reshape(1:12, 3, 4)))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y = trainmode(m, x)
        cy = trainmode(cm, cx)

        @test cy isa CuArray{Float32,2}

        @test cpu(cy) ≈ y

        g = gradient(()->sum(m(x)), params(m))
        cg = gradient(()->sum(cm(cx)), params(cm))

        @test g[m.γ] ≈ cpu(cg[cm.γ])
        @test g[m.β] ≈ cpu(cg[cm.β])
    end
end
