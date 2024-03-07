using Test
using Flux
using Enzyme
using EnzymeTestUtils
using Functors
# using EnzymeCore

make_zero(x::AbstractArray) = zero(x)
make_zero(x::Number) = zero(x)
make_zero(x) = x

make_differential(model) = fmap(make_zero, model)

function grad(f, x...)
    args = []
    for x in x
        if x isa Number
            push!(args, Active(x))
        else
            push!(args, Duplicated(x, make_differential(x)))
        end
    end
    @show x args
    ret = Enzyme.autodiff(ReverseWithPrimal, f, Active, args...)
    g = ntuple(i -> x[i] isa Number ? ret[1][i] : args[i].dval, length(x))
    return g
end

@testset "grad" begin
    @testset "number and arrays" begin
        f(x, y) = sum(x.^2) + y^3
        x = Float32[1, 2, 3]
        y = 3f0
        g = grad(f, x, y)
        @test g[1] isa Array{Float32}
        @test g[2] isa Float32
        @test g[1] ≈ 2x
        @test g[2] ≈ 3*y^2
    end

    @testset "struct" begin
        struct SimpleDense{W, B, F}
            weight::W
            bias::B
            σ::F
        end
        SimpleDense(in::Integer, out::Integer; σ=identity) = SimpleDense(randn(Float32, out, in), zeros(Float32, out), σ)
        (m::SimpleDense)(x) = m.σ.(m.weight * x .+ m.bias)
        @functor SimpleDense

        model = SimpleDense(2, 4)
        x = randn(Float32, 2)
        loss(model, x) = sum(model(x))

        g = grad(loss, model, x)
        @test g[1] isa SimpleDense
        @test g[2] isa Array{Float32}
        @test g[1].weight isa Array{Float32}
        @test g[1].bias isa Array{Float32}
        @test g[1].weight ≈ ones(Float32, 4, 1) .* x'
        @test g[1].bias ≈ ones(Float32, 4)
    end
end

