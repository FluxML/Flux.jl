@test Flux.AMDGPU_LOADED[]

# @testset "Basic GPU movement" begin
#     @test Flux.amd(rand(Float64, 16)) isa ROCArray{Float32, 1}
#     @test Flux.amd(rand(Float64, 16, 16)) isa ROCArray{Float32, 2}
#     @test Flux.amd(rand(Float32, 16, 16)) isa ROCArray{Float32, 2}
#     @test Flux.amd(rand(Float16, 16, 16, 16)) isa ROCArray{Float16, 3}

#     @test gradient(x -> sum(Flux.amd(x)), rand(Float32, 4, 4)) isa Tuple
#     @test gradient(x -> sum(cpu(x)), AMDGPU.rand(Float32, 4, 4)) isa Tuple
# end

# @testset "Dense no bias" begin
#     m = Dense(3 => 4; bias=false) |> Flux.amd
#     x = zeros(Float32, 3, 4) |> Flux.amd
#     @test sum(m(x)) ≈ 0f0
#     gs = gradient(m -> sum(m(x)), m)
#     @test isnothing(gs[1].bias)
# end

# @testset "Chain of Dense layers" begin
#     m = Chain(Dense(10, 5, tanh), Dense(5, 2), softmax) |> f32
#     x = rand(Float32, 10, 10)
#     amdgputest(m, x)
# end

@testset "Convolution" begin
    m = Conv((2, 2), 1 => 1) |> f32
    x = rand(Float32, 4, 4, 1, 1)
    amdgputest(m, x; atol=1f-3, checkgrad=false)

    # Gradients are flipped as well.
    md, xd = Flux.amd.((m, x))
    gs = gradient(m -> sum(m(x)), m)
    gsd = gradient(m -> sum(m(xd)), md)
    @test gs[1].weight[end:-1:1, end:-1:1, :, :] ≈ Array(gsd[1].weight) atol=1f-3
end

# @testset "Cross-correlation" begin
#     m = CrossCor((2, 2), 3 => 4) |> f32
#     x = rand(Float32, 10, 10, 3, 2)
#     amdgputest(m, x; atol=1f-3)
# end

# @testset "Restructure" begin
#     m = Dense(1, 1) |> Flux.amd
#     θ, m̂ = Flux.destructure(m)
#     foo(x) = sum(re(p)(x))

#     x = Flux.amd(rand(Float32, 1))
#     @test gradient(x -> sum(m̂(θ)(x)), x)[1] isa ROCArray{Float32}
# end

# @testset "Flux.amd(x) on structured arrays" begin
#     g1 = Zygote.OneElement(1, (2, 3), axes(ones(4, 5)))
#     @test Flux.amd(g1) isa ROCMatrix{Int64}
#     g2 = Zygote.Fill(1f0, 2)
#     @test Flux.amd(g2) isa ROCArray{Float32, 1}
#     g3 = transpose(Float32[1 2; 3 4])
#     @test parent(Flux.amd(g3)) isa ROCMatrix{Float32}
# end

# @testset "Flux.onecold gpu" begin
#     y = Flux.onehotbatch(ones(3), 1:10) |> Flux.amd
#     l = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
#     @test Flux.onecold(y) isa ROCArray
#     @test y[3, :] isa ROCArray
#     @test Flux.onecold(y, l) == ['a', 'a', 'a']
# end

# FIXME scalar indexing. Needs NNlib.scatter?
# @testset "Flux.onehot gpu" begin
#     y = Flux.onehotbatch(ones(3), 1:2) |> Flux.amd
#     x = rand(3, 2) |> Flux.amd
#     @test gradient(x -> sum(x * y), x)[1] isa ROCArray
# end
