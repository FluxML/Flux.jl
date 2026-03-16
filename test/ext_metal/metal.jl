@testset "data movement" begin
    metal_device = Flux.gpu_device()
    cdev = cpu_device()

    @test metal_device isa Flux.MetalDevice

    x = randn(Float32, 5, 5)
    cx = x |> metal_device
    @test cx isa Metal.MtlMatrix{Float32}
    x2 = cx |> cdev
    @test x2 isa Matrix{Float32}
    @test x ≈ x2
    
    metal_device = gpu_device(1)
    @test metal_device isa Flux.MetalDevice

    @test cpu(cx) isa Matrix{Float32}
    @test cpu(cx) ≈ x

    @test gpu(x) isa Metal.MtlMatrix{Float32}
    @test cpu(gpu(x)) ≈ x
end

@testset "Basic GPU movement" begin
    @test Flux.gpu(rand(Float64, 16)) isa MtlArray{Float32, 1}
    @test Flux.gpu(rand(Float64, 16, 16)) isa MtlArray{Float32, 2}
    @test Flux.gpu(rand(Float32, 16, 16)) isa MtlArray{Float32, 2}
    @test Flux.gpu(rand(Float16, 16, 16, 16)) isa MtlArray{Float16, 3}

    @test gradient(x -> sum(Flux.gpu(x)), rand(Float32, 4, 4)) isa Tuple
    @test gradient(x -> sum(cpu(x)), Metal.rand(Float32, 4, 4)) isa Tuple
end

@testset "Dense no bias" begin
    m = Dense(3 => 4; bias=false) |> Flux.gpu
    x = zeros(Float32, 3, 4) |> Flux.gpu
    @test m(x) isa MtlArray{Float32, 2}
    @test sum(m(x)) ≈ 0f0
    gs = gradient(m -> sum(m(x)), m)
    @test isnothing(gs[1].bias)
end

@testset "Chain of Dense layers" begin
    m = Chain(Dense(10, 5, tanh), Dense(5, 2), softmax)
    x = rand(Float32, 10, 10) 
    @test (m|>gpu)(x|>gpu) isa MtlArray{Float32, 2}
    test_gradients(m, x, test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=nothing)
end

@testset "gradients" begin
    broken_models = ["Conv", "Chain(Conv, Conv)", "Chain(Conv, MeanPool)", "ConvTranspose","Bilinear","MultiHeadAttention"]
    # Bilinear and MultiHeadAttention will be fixed by https://github.com/FluxML/NNlib.jl/pull/614
    for (model, x, name) in TEST_MODELS
        @testset "Zygote grad check $name" begin
            @test test_gradients(model, x; test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=nothing) broken=(name ∈ broken_models)
        end
    end
end

@testset "Recurrent" begin
    global BROKEN_TESTS = [:lstm, :gru, :gruv3]
    include("../test_common/gpu_recurrent.jl")
end

@testset "Normalization" begin
    include("../test_common/normalization.jl")
    normalization_testsuite(Flux.gpu_device(force=true))
end

@testset "Huber Loss test" begin
    X = Flux.gpu(Float32[0,1])
    Y = Flux.gpu(Float32[1,0])

    grad = Flux.gradient(X, Y) do a,b
        Flux.Losses.huber_loss(a,b)
    end

    @test Flux.cpu(grad[1]) == [-0.5, 0.5]
    @test Flux.cpu(grad[2]) == [0.5, -0.5]
end
