using Flux, Test


@testset "attention" begin
    dim = 4; nheads = 2; len = 3; batch_size = 5
    mha = MultiHeadAttention(dim, nheads)
    q = rand(Float32, (dim, len, batch_size))
    k = rand(Float32, (dim, len, batch_size))
    v = rand(Float32, (dim, len, batch_size))
    
    y, α = mha(q, k, v, withscores=true)
    @test y isa Array{Float32, 3}
    @test size(y) == (dim, len, batch_size)
    @test α isa Array{Float32, 4}
    @test size(α) == (len, len, nheads, batch_size)

    @testset "self-attention" begin
        y1 = mha(q)
        y2 = mha(q, q, q)
        @test y1 ≈ y2
    end

    @testset "key and value are the same" begin
        y1 = mha(q, k)
        y2 = mha(q, k, k)
        @test y1 ≈ y2
    end

    @testset "change dims" begin
        dims = 4 => 10 => 5
        nhead = 5
        mha2 = MultiHeadAttention(dims, nheads)
        y2 = mha2(q, k, v)
        @test size(y2) == (dims.second.second, len, batch_size)
    end
end

