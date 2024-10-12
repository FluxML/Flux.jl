

@testset "attention" begin
  dim = 4; nheads = 2; len = 3; batch_size = 5
  mha = MultiHeadAttention(dim; nheads)
  q = rand(Float32, (dim, len, batch_size))
  k = rand(Float32, (dim, len, batch_size))
  v = rand(Float32, (dim, len, batch_size))
  
  y, α = mha(q, k, v)
  @test y isa Array{Float32, 3}
  @test size(y) == (dim, len, batch_size)
  @test α isa Array{Float32, 4}
  @test size(α) == (len, len, nheads, batch_size)

  @testset "self-attention" begin
    y1, α1 = mha(q)
    y2, α2 = mha(q, q, q)
    @test y1 ≈ y2
    @test α1 ≈ α2
  end

  @testset "key and value are the same" begin
    y1, α1 = mha(q, k)
    y2, α2 = mha(q, k, k)
    @test y1 ≈ y2
    @test α1 ≈ α2
  end

  @testset "change dims" begin
    dims = 4 => 10 => 5
    nhead = 5
    mha2 = MultiHeadAttention(dims; nheads)
    y2, _ = mha2(q, k, v)
    @test size(y2) == (dims.second.second, len, batch_size)
  end

  @testset "mask" begin
    mask = NNlib.make_causal_mask(q)
    y, α = mha(q; mask)
    @test all(α[2, 1, :, :] .== 0)
    @test α[:, :, 1, 1] ≈ triu(α[:, :, 1, 1])
  end

  @testset "bias" begin
    # use bias to produce a causal mask
    b = zeros(Float32, (len, len))
    for i in 1:len, j in i:len
      b[i, j] = typemax(Float32)
    end
    y, α = mha(q, k, v, b)
    @test all(α[2, 1, :, :] .== 0)
    @test α[:, :, 1, 1] ≈ triu(α[:, :, 1, 1])  
  end

  @testset "gradient" begin
    test_gradients(mha, q, loss = ((y, α)) -> sum(y.^2) + sum(α.^2))
  end
end

