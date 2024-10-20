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
    m = Chain(Dense(10, 5, tanh), Dense(5, 2), softmax)
    x = rand(Float32, 10, 10)
    test_gradients(m, x, test_gpu=true, compare_finite_diff=false)
end

@testset "Convolution" begin
    for conv_type in (Conv, ConvTranspose), nd in 1:3
        m = conv_type(tuple(fill(2, nd)...), 3 => 4)
        x = rand(Float32, fill(10, nd)..., 3, 5)

        md, xd = Flux.gpu.((m, x))
        y = m(x)
        # Ensure outputs are the same.
        @test collect(md(xd)) ≈ y  atol=1f-3

        # Gradients are flipped as well.
        gs = gradient(m -> sum(m(x)), m)[1]
        gsd = gradient(m -> sum(m(xd)), md)[1]

        dims = ntuple(i -> i, ndims(m.weight) - 2)
        @test reverse(gs.weight; dims) ≈ Array(gsd.weight) atol=1f-2

        # Movement back to CPU flips weights back.
        mh = Flux.cpu(md)
        @test m.weight ≈ mh.weight
    end
end

@testset "Convolution with symmetric non-constant padding" begin
    for conv_type in (Conv, ConvTranspose), nd in 1:3
        kernel = tuple(fill(2, nd)...)
        x = rand(Float32, fill(10, nd)..., 3, 5) |> gpu

        pad = ntuple(i -> i, nd)
        m = conv_type(kernel, 3 => 4, pad=pad) |> gpu

        expanded_pad = ntuple(i -> pad[(i - 1) ÷ 2 + 1], 2 * nd)
        m_expanded = conv_type(kernel, 3 => 4, pad=expanded_pad) |> gpu

        @test size(m(x)) == size(m_expanded(x))
    end
end

@testset "ConvTranspose output padding" begin
    x = randn(Float32, 10, 11, 3, 2)
    m = ConvTranspose((3, 5), 3=>6, stride=3, outpad=(1, 0))
    md, xd = Flux.gpu.((m, x))
    @test size(m(x)) == size(md(xd))

    x = randn(Float32, 10, 11, 12, 3, 2)
    m = ConvTranspose((3, 5, 3), 3=>6, stride=3, outpad=(1, 0, 1))
    md, xd = Flux.gpu.((m, x))
    @test size(m(x)) == size(md(xd))
end

@testset "Chain(Conv)" begin
    m = Chain(Conv((3, 3), 3 => 3))
    x = rand(Float32, 5, 5, 3, 2)
    test_gradients(m, x, test_gpu=true, compare_finite_diff=false, test_grad_f=false)

    md = m |> gpu |> cpu
    @test md[1].weight ≈ m[1].weight atol=1f-3

    m = Chain(ConvTranspose((3, 3), 3 => 3))
    x = rand(Float32, 5, 5, 3, 2)
    test_gradients(m, x, test_gpu=true, compare_finite_diff=false, test_grad_f=false)

    md = m |> gpu |> cpu
    @test md[1].weight ≈ m[1].weight atol=1f-3
end

@testset "Cross-correlation" begin
    m = CrossCor((2, 2), 3 => 4)
    x = rand(Float32, 5, 5, 3, 2)
    test_gradients(m, x, test_gpu=true, compare_finite_diff=false)
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

@testset "cpu and gpu on RNGs" begin
    crng = Random.default_rng()
    grng = gpu(crng)
    @test grng isa AMDGPU.rocRAND.RNG
    @test cpu(grng) === crng
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
        test_gradients(bn, x; test_gpu=true, compare_finite_diff=false)
    end
end

@testset "gpu(::DataLoader)" begin
    X = randn(Float64, 3, 33)
    pre1 = Flux.DataLoader(X |> Flux.gpu; batchsize=13, shuffle=false)
    post1 = Flux.DataLoader(X; batchsize=13, shuffle=false) |> Flux.gpu
    for epoch in 1:2
        for (p, q) in zip(pre1, post1)
            @test p isa ROCArray{Float32}
            @test q isa ROCArray{Float32}
            @test p ≈ q
        end
    end
end
