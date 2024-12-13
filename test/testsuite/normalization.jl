function normalization_testsuite(dev)
    @testset "WeightNorm" begin
        x = rand(Float32, 1, 3) |> dev
        mn = WeightNorm(Dense(1 => 2)) |> dev
        m = Flux.remove_weight_norms(mn)
        @test m(x) ≈ mn(x)

        @test_throws ArgumentError WeightNorm(m, :weights)
        @test_throws "does not have field" WeightNorm(m, :weights)

        @test_throws ArgumentError WeightNorm(m, :bias)
        @test_throws "is all zero" WeightNorm(m, :bias)

        og = (Zygote.gradient(m) do m
            sum(m(x))
        end)[1]
        g = (Zygote.gradient(mn) do mn
            sum(mn(x))
        end)[1]

        @test g.layer.weight ≢ nothing # Original weight acts as a direction `v`.
        @test g.layer.bias ≢ nothing
        @test g.g ≢ nothing

        # Compare gradients with original layer.

        v = mn.layer.weight
        ϵ = eps(eltype(v))
        n2 = sum(abs2, v; dims=2)
        v = v ./ sqrt.(n2 .+ ϵ)

        @test (og.weight .* v) ≈ g.g
        @test (og.weight .* mn.g .- mn.g .* g.g .* v) ≈ g.layer.weight atol=1f-6

        # Test WeightNorm removal.

        om = Flux.remove_weight_norms(mn)
        @test om isa Dense
        @test om.weight ≈ m.weight
        @test om.bias ≈ m.bias

        # Test with Chain.

        c = Chain(
            WeightNorm(Conv((3,), 1 => 2)),
            Conv((3,), 2 => 2),
            WeightNorm(Conv((3,), 2 => 3)),
            x -> reshape(x, 18, :),
            WeightNorm(Dense(18, 4)),
            Dense(4, 1),
        )
        @test c[1] isa WeightNorm
        @test c[2] isa Conv
        @test c[3] isa WeightNorm
        @test c[5] isa WeightNorm
        @test c[6] isa Dense

        oc = Flux.remove_weight_norms(c)
        @test oc[1] isa Conv
        @test oc[2] isa Conv
        @test oc[3] isa Conv
        @test oc[5] isa Dense
        @test oc[6] isa Dense

        x = rand(Float32, 12, 1, 1)
        @test c(x) ≈ oc(x)
    end
end
