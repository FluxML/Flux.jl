using Test
using Flux
using Enzyme
using EnzymeTestUtils
using Functors
# using EnzymeCore

Enzyme.API.runtimeActivity!(true) # for Enzyme debugging 

make_zero(x::Union{Number,AbstractArray}) = zero(x)
make_zero(x) = x
make_differential(model) = fmap(make_zero, model)
# make_differential(model) = fmapstructure(make_zero, model) # NOT SUPPORTED, See https://github.com/EnzymeAD/Enzyme.jl/issues/1329

function grad(f, x...)
    args = []
    for x in x
        if x isa Number
            push!(args, Active(x))
        else
            push!(args, Duplicated(x, make_differential(x)))
        end
    end
    # @show x args
    ret = Enzyme.autodiff(ReverseWithPrimal, f, Active, args...)
    g = ntuple(i -> x[i] isa Number ? ret[1][i] : args[i].dval, length(x))
    return g
end

function check_grad(g1, g2; broken=false)
    fmap(g1, g2) do x, y
        if x isa Union{Number, AbstractArray{<:Number}}
            # @test y isa typeof(x)
            # @show x y
            @test x ≈ y rtol=1e-4 atol=1e-4 broken=broken
        end
        return x
    end
end

function test_enzyme_grad(model, x)
    loss(model, x) = sum(model(x))
    
    Flux.reset!(model)
    l = loss(model, x)
    Flux.reset!(model)
    @test loss(model, x) == l # Check loss doesn't change with multiple runs


    Flux.reset!(model)
    grads_flux = Flux.gradient(loss, model, x)

    Flux.reset!(model)
    grads_enzyme = grad(loss, model, x)

    check_grad(grads_flux, grads_enzyme)
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

@testset "Models" begin
    models_xs = [
        (Dense(2, 4), randn(Float32, 2), "Dense"),
        (Chain(Dense(2, 4, relu), Dense(4, 3)), randn(Float32, 2), "Chain(Dense, Dense)"),
        (f64(Chain(Dense(2, 4), Dense(4, 2))), randn(Float64, 2, 1), "f64(Chain(Dense, Dense))"),
        (Flux.Scale([1.0f0 2.0f0 3.0f0 4.0f0], true, abs2), randn(Float32, 2), "Flux.Scale"),
        (Conv((3, 3), 2 => 3), randn(Float32, 3, 3, 2, 1), "Conv"),
        (Chain(Conv((3, 3), 2 => 3, relu), Conv((3, 3), 3 => 1, relu)), rand(Float32, 5, 5, 2, 1), "Chain(Conv, Conv)"),
        (Chain(Conv((5, 5), 3 => 7, pad=SamePad()), MaxPool((5, 5), pad=SamePad())), rand(Float32, 100, 100, 3, 50), "Chain(Conv, MaxPool)"),
        (Maxout(() -> Dense(5 => 7, tanh), 3), randn(Float32, 5, 1), "Maxout"),
        # BROKEN, uncomment as tests below are fixed
        # (RNN(3 => 5), randn(Float32, 3, 10), "RNN"), 
        # (Chain(RNN(3 => 5), RNN(5 => 3)), randn(Float32, 3, 10), "Chain(RNN, RNN)"), # uncomment when broken test below is fixed
        # (LSTM(3 => 5), randn(Float32, 3, 10), "LSTM"),
        # (Chain(LSTM(3 => 5), LSTM(5 => 3)), randn(Float32, 3, 10), "Chain(LSTM, LSTM)"),
    ]
    
    for (model, x, name) in models_xs
        @testset "check grad $name" begin
            test_enzyme_grad(model, x)
        end
    end   
end

@testset "Broken Models" begin
    loss(model, x) = sum(model(x))
    
    @testset "RNN" begin
        model = RNN(3 => 5)
        x = randn(Float32, 3, 10)
        Flux.reset!(model)
        grads_flux = Flux.gradient(loss, model, x)
        Flux.reset!(model)
        grads_enzyme = grad(loss, model, x)
        check_grad(grads_flux[1].state, grads_enzyme[1].state, broken=true)
    end

    @testset "LSTM" begin
        model = LSTM(3 => 5)
        x = randn(Float32, 3, 10)
        Flux.reset!(model)
        grads_flux = Flux.gradient(loss, model, x)
        Flux.reset!(model)
        grads_enzyme = grad(loss, model, x)
        check_grad(grads_flux[1].state, grads_enzyme[1].state, broken=true)
    end
end
