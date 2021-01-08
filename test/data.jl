@testset "DataLoader" begin
    X = reshape([1:10;], (2, 5))
    Y = [1:5;]

    d = DataLoader(X, batchsize=2)
    @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == typeof(X)
    @test length(batches) == 3
    @test batches[1] == X[:,1:2]
    @test batches[2] == X[:,3:4]
    @test batches[3] == X[:,5:5]

    d = DataLoader(X, batchsize=2, partial=false)
    @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == typeof(X)
    @test length(batches) == 2
    @test batches[1] == X[:,1:2]
    @test batches[2] == X[:,3:4]

    d = DataLoader((X,), batchsize=2, partial=false)
    @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == Tuple{typeof(X)}
    @test length(batches) == 2
    @test batches[1] == (X[:,1:2],)
    @test batches[2] == (X[:,3:4],)

    d = DataLoader((X, Y), batchsize=2)
    @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == Tuple{typeof(X), typeof(Y)}
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

    # test with NamedTuple
    d = DataLoader((x=X, y=Y), batchsize=2)
    @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == NamedTuple{(:x, :y), Tuple{typeof(X), typeof(Y)}}
    @test length(batches) == 3
    @test length(batches[1]) == 2
    @test length(batches[2]) == 2
    @test length(batches[3]) == 2
    @test batches[1][1] == batches[1].x == X[:,1:2]
    @test batches[1][2] == batches[1].y == Y[1:2]
    @test batches[2][1] == batches[2].x == X[:,3:4]
    @test batches[2][2] == batches[2].y == Y[3:4]
    @test batches[3][1] == batches[3].x == X[:,5:5]
    @test batches[3][2] == batches[3].y == Y[5:5]

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
    d  = DataLoader((X, Y))
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

@testset "Tree show" begin
    # testcase for issue #1354
    # testing that methods(Base.show) does not throw. Having something more specific would be too fragile
    buf = IOBuffer()
    Base.show(buf, filter(x->x.module == Flux, methods(Base.show).ms))
    str_repr = String(take!(buf))
    @test !isempty(str_repr)
end