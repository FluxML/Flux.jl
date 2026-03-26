@testset "AutoADTypes gradient" begin
    m = Dense(2 => 2)
    x = rand(Float32, 2)
    g_zygote = Flux.gradient(m -> sum(m(x)), AutoZygote(), m)[1]
    g_mooncake = Flux.gradient(m -> sum(m(x)), AutoMooncake(), m)[1]
    @test g_zygote.weight ≈ g_mooncake.weight
end
