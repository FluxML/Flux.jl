@test Flux.AMDGPU_LOADED[]

@testset "Basic GPU movement" begin
    @test Flux.gpu(rand(Float64, 16)) isa ROCArray{Float32, 1}
    @test Flux.gpu(rand(Float64, 16, 16)) isa ROCArray{Float32, 2}
    @test Flux.gpu(rand(Float32, 16, 16)) isa ROCArray{Float32, 2}
    @test Flux.gpu(rand(Float16, 16, 16, 16)) isa ROCArray{Float16, 3}

    @test gradient(x -> sum(Flux.gpu(x)), rand(Float32, 4, 4)) isa Tuple
    @test gradient(x -> sum(cpu(x)), AMDGPU.rand(Float32, 4, 4)) isa Tuple
end

@testset "Dense no bias" begin
    m = Dense(3 => 4; bias=false) |> Flux.gpu
    x = zeros(Float32, 3, 4) |> Flux.gpu
    @test sum(m(x)) ≈ 0f0
    gs = gradient(m -> sum(m(x)), m)
    @test isnothing(gs[1].bias)
end

@testset "Chain of Dense layers" begin
    m = Chain(Dense(10, 5, tanh), Dense(5, 2), softmax) |> f32
    x = rand(Float32, 10, 10)
    amdgputest(m, x)
end

@testset "Convolution" begin
    for nd in 1:3
        m = Conv(tuple(fill(2, nd)...), 3 => 4) |> f32
        x = rand(Float32, fill(10, nd)..., 3, 5)

        # Ensure outputs are the same.
        amdgputest(m, x; atol=1f-3, checkgrad=false)

        # Gradients are flipped as well.
        md, xd = Flux.gpu.((m, x))
        gs = gradient(m -> sum(m(x)), m)
        gsd = gradient(m -> sum(m(xd)), md)

        dims = ntuple(i -> i, ndims(m.weight) - 2)
        @test reverse(gs[1].weight; dims) ≈ Array(gsd[1].weight) atol=1f-2

        # Movement back to CPU flips weights back.
        mh = Flux.cpu(md)
        @test m.weight ≈ mh.weight
    end
end

@testset "Cross-correlation" begin
    m = CrossCor((2, 2), 3 => 4) |> f32
    x = rand(Float32, 10, 10, 3, 2)
    amdgputest(m, x; atol=1f-3)
end

@testset "Restructure" begin
    m = Dense(1, 1) |> Flux.gpu
    θ, m̂ = Flux.destructure(m)
    foo(x) = sum(re(p)(x))

    x = Flux.gpu(rand(Float32, 1))
    @test gradient(x -> sum(m̂(θ)(x)), x)[1] isa ROCArray{Float32}
end

@testset "Flux.gpu(x) on structured arrays" begin
    g1 = Zygote.OneElement(1, (2, 3), axes(ones(4, 5)))
    @test Flux.gpu(g1) isa ROCMatrix{Int64}
    g2 = Zygote.Fill(1f0, 2)
    @test Flux.gpu(g2) isa ROCArray{Float32, 1}
    g3 = transpose(Float32[1 2; 3 4])
    @test parent(Flux.gpu(g3)) isa ROCMatrix{Float32}
end

@testset "Flux.onecold gpu" begin
    y = Flux.onehotbatch(ones(3), 1:10) |> Flux.gpu
    l = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    @test Flux.onecold(y) isa ROCArray
    @test y[3, :] isa ROCArray
    @test Flux.onecold(y, l) == ['a', 'a', 'a']
end

@testset "Batchnorm" begin
    bn = BatchNorm(3, σ)
    for nd in 1:3
        x = rand(Float32, fill(2, nd - 1)..., 3, 4)
        amdgputest(bn, x; atol=1f-3, allow_nothing=true)
    end
end

# FIXME scalar indexing. Needs NNlib.scatter?
# @testset "Flux.onehot gpu" begin
#     y = Flux.onehotbatch(ones(3), 1:2) |> Flux.gpu
#     x = rand(3, 2) |> Flux.gpu
#     @test gradient(x -> sum(x * y), x)[1] isa ROCArray
# end
