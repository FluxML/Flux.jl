@testset "DataLoader" begin
    X = reshape([1:10;], (2, 5))
    Y = [1:5;]

    d = DataLoader(X, batchsize=2)
    batches = collect(d)
    @test length(batches) == 3
    @test batches[1] == X[:,1:2]
    @test batches[2] == X[:,3:4]
    @test batches[3] == X[:,5:5]

    d = DataLoader(X, batchsize=2, partial=false)
    batches = collect(d)
    @test length(batches) == 2
    @test batches[1] == X[:,1:2]
    @test batches[2] == X[:,3:4]

    d = DataLoader(X, Y, batchsize=2)
    batches = collect(d)
    @test length(batches) == 3
    @test length(batches[1]) == 2
    @test length(batches[2]) == 2
    @test length(batches[3]) == 2
    @test batches[1][1] == X[:,1:2]
    @test batches[1][2] == Y[1:2]
    @test batches[2][1] == X[:,3:4]
    @test batches[2][2] == Y[3:4]
    @test batches[3][1] == X[:,5:5]
    @test batches[3][2] == Y[5:5]

    # test interaction with `train!`
    θ = ones(2)
    X = zeros(2, 10)
    loss(x) = sum((x .- θ).^2)
    d  = DataLoader(X) 
    Flux.train!(loss, [θ], ncycle(d, 10), Descent(0.1))
    @test norm(θ) < 1e-4

    # test interaction with `train!`
    θ = zeros(2)
    X = ones(2, 10)
    Y = fill(2, 10)
    loss(x, y) = sum((y - x'*θ).^2)
    d  = DataLoader(X, Y) 
    Flux.train!(loss, [θ], ncycle(d, 10), Descent(0.1))
    @test norm(θ .- 1) < 1e-10
end

@testset "CMUDict" begin 
    @test cmudict()["CATASTROPHE"] == :[K,AH0,T,AE1,S,T,R,AH0,F,IY0].args

    @test length(CMUDict.phones()) == 39

    @test length(CMUDict.symbols()) == 84
end

@testset "MNIST" begin 
    @test MNIST.images()[1] isa Matrix
    @test MNIST.labels() isa Vector{Int64}
end

@testset "FashionMNIST" begin 
    @test FashionMNIST.images()[1] isa Matrix
    @test FashionMNIST.labels() isa Vector{Int64}
end

@testset "Sentiment" begin 
    @test Data.Sentiment.train() isa Vector{Data.Tree{Any}}
end

@testset "Iris" begin 
    @test Iris.features() isa Matrix
    @test size(Iris.features()) == (4,150)

    @test Iris.labels() isa Vector{String}
    @test size(Iris.labels()) == (150,)
end


@testset "Housing" begin
    @test Housing.features() isa Matrix # test broken due to SSL certifate expiration problem
    @test size(Housing.features()) == (506, 13)

    @test Housing.targets() isa Array{Float64}
    @test size(Housing.targets()) == (506, 1)
end
