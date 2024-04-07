using LinearAlgebra


@testset "RNN gradients" begin
    layer = Flux.Recur(Flux.RNNCell(1 => 1, identity))
    layer.cell.Wi .= 5.0f0
    layer.cell.Wh .= 4.0f0
    layer.cell.b .= 0.0f0
    layer.cell.state0 .= 7.0f0
    x = [[2.0f0], [3.0f0]]

    # theoretical primal gradients
    primal =
        layer.cell.Wh .* (layer.cell.Wh * layer.cell.state0 .+ x[1] .* layer.cell.Wi) .+
        x[2] .* layer.cell.Wi
    ∇Wi = x[1] .* layer.cell.Wh .+ x[2]
    ∇Wh = 2 .* layer.cell.Wh .* layer.cell.state0 .+ x[1] .* layer.cell.Wi
    ∇b = layer.cell.Wh .+ 1
    ∇state0 = layer.cell.Wh .^ 2

    Flux.reset!(layer)
    e, g = Flux.withgradient(layer) do m
        out = [m(xi) for xi in x]
        sum(out[2])
    end
    grads = g[1][:cell]

    @test primal[1] ≈ e

    @test_broken ∇Wi ≈ grads[:Wi]
    @test_broken ∇Wh ≈ grads[:Wh]
    @test_broken ∇b ≈ grads[:b]
    @test_broken ∇state0 ≈ grads[:state0]
end

# Ref FluxML/Flux.jl#1209 1D input
@testset "BPTT-1D" begin
  seq = [rand(Float32, 2) for i = 1:3]
  for r ∈ [RNN,]
    rnn = r(2 => 3)
    Flux.reset!(rnn)
    grads_seq = gradient(rnn) do rnn
        sum([rnn(s) for s in seq][3])
    end[1]

    Flux.reset!(rnn);
    bptt = gradient(Wh -> sum(tanh.(rnn.cell.Wi * seq[3] + Wh *
                                  tanh.(rnn.cell.Wi * seq[2] + Wh *
                                        tanh.(rnn.cell.Wi * seq[1] +
                                            Wh * rnn.cell.state0
                                        + rnn.cell.b)
                                  + rnn.cell.b)
                            + rnn.cell.b)),
                    rnn.cell.Wh)
    @test grads_seq.cell.Wh ≈ bptt[1]
  end
end

# Ref FluxML/Flux.jl#1209 2D input
@testset "BPTT-2D" begin
  seq = [rand(Float32, (2, 1)) for i = 1:3]
  for r ∈ [RNN,]
    rnn = r(2 => 3)
    Flux.reset!(rnn)
    grads_seq = gradient(rnn) do rnn
        sum([rnn(s) for s in seq][3])
    end[1]
    Flux.reset!(rnn);
    bptt = gradient(Wh -> sum(tanh.(rnn.cell.Wi * seq[3] + Wh *
                                  tanh.(rnn.cell.Wi * seq[2] + Wh *
                                        tanh.(rnn.cell.Wi * seq[1] +
                                            Wh * rnn.cell.state0
                                        + rnn.cell.b)
                                  + rnn.cell.b)
                            + rnn.cell.b)),
                    rnn.cell.Wh)
    @test grads_seq.cell.Wh ≈ bptt[1]
  end
end

@testset "BPTT-3D" begin
  seq = rand(Float32, (2, 1, 3))
  rnn = RNN(2 => 3)
  Flux.reset!(rnn)
  grads_seq = gradient(rnn) do rnn
    sum(rnn(seq)[:, :, 3])
  end[1]
  Flux.reset!(rnn);
  bptt = gradient(rnn.cell.Wh) do Wh
    # calculate state 1
    s1 = tanh.(rnn.cell.Wi * seq[:, :, 1] +
               Wh * rnn.cell.state0 +
               rnn.cell.b)
    #calculate state 2
    s2 = tanh.(rnn.cell.Wi * seq[:, :, 2] +
               Wh * s1 +
               rnn.cell.b)
    #calculate state 3
    s3 = tanh.(rnn.cell.Wi * seq[:, :, 3] +
               Wh * s2 +
               rnn.cell.b)
    sum(s3) # loss is sum of state 3
  end
  @test grads_seq.cell.Wh ≈ bptt[1]
end

@testset "RNN-shapes" begin
  @testset for R in [RNN, GRU, LSTM, GRUv3]
    m1 = R(3 => 5)
    m2 = R(3 => 5)
    x1 = rand(Float32, 3)
    x2 = rand(Float32, 3, 1)
    x3 = rand(Float32, 3, 1, 2)
    Flux.reset!(m1)
    Flux.reset!(m2)
    @test size(m1(x1)) == (5,)
    @test size(m1(x1)) == (5,) # repeat in case of effect from change in state shape
    @test size(m2(x2)) == (5, 1)
    @test size(m2(x2)) == (5, 1)
  end
end

@testset "multigate" begin
  x = rand(6, 5)
  res, (dx,) = Flux.withgradient(x) do x
    x1, _, x3 = Flux.multigate(x, 2, Val(3))
    sum(x1) + sum(x3 .* 2)
  end
  @test res == sum(x[1:2, :]) + 2sum(x[5:6, :])
  @test dx == [ones(2, 5); zeros(2, 5); fill(2, 2, 5)]
end

@testset "eachlastdim" begin
  x = rand(3, 3, 1, 2, 4)
  @test length(Flux.eachlastdim(x)) == size(x, ndims(x))
  @test collect(@inferred(Flux.eachlastdim(x))) == collect(eachslice(x; dims=ndims(x)))
  slicedim = (size(x)[1:end-1]..., 1)
  res, (dx,) = Flux.withgradient(x) do x
    x1, _, x3, _ = Flux.eachlastdim(x)
    sum(x1) + sum(x3 .* 3)
  end
  @test res ≈ sum(selectdim(x, ndims(x), 1)) + 3sum(selectdim(x, ndims(x), 3))
  @test dx ≈ cat(fill(1, slicedim), fill(0, slicedim),
              fill(3, slicedim), fill(0, slicedim); dims=ndims(x))
end

@testset "∇eachlastdim" begin
    x = rand(3, 3, 1, 2, 4)
    x_size = size(x)
    y = collect(eachslice(x; dims=ndims(x)))
    @test @inferred(Flux.∇eachlastdim(y, x)) == x
    ZeroTangent = Flux.Zygote.ZeroTangent
    NoTangent = Flux.Zygote.NoTangent
    abstract_zeros_vector = [ZeroTangent(), ZeroTangent(), NoTangent(), NoTangent()]
    @test @inferred(Flux.∇eachlastdim(abstract_zeros_vector, x)) == zeros(size(x))
    x2 = rand(Float64, x_size[1:end-1])
    x3 = rand(Float64, x_size[1:end-1])
    mixed_vector = [ZeroTangent(), x2, x3, ZeroTangent()]
    @test @inferred(Flux.∇eachlastdim(mixed_vector, x)) ≈ cat(zeros(x_size[1:end-1]), 
                                                         x2, 
                                                         x3, 
                                                         zeros(x_size[1:end-1]); dims=ndims(x))
end

@testset "Different Internal Matrix Types" begin
  R = Flux.Recur(Flux.RNNCell(tanh, rand(5, 3), Tridiagonal(rand(5, 5)), rand(5), rand(5, 1)))
  # don't want to pull in SparseArrays just for this test, but there aren't any
  # non-square structured matrix types in LinearAlgebra. so we will use a different
  # eltype matrix, which would fail before when `W_i` and `W_h` were required to be the
  # same type.
  L = Flux.Recur(Flux.LSTMCell(rand(5*4, 3), rand(1:20, 5*4, 5), rand(5*4), (rand(5, 1), rand(5, 1))))
  G = Flux.Recur(Flux.GRUCell(rand(5*3, 3), rand(1:20, 5*3, 5), rand(5*3), rand(5, 1)))
  G3 = Flux.Recur(Flux.GRUv3Cell(rand(5*3, 3), rand(1:20, 5*2, 5), rand(5*3), Tridiagonal(rand(5, 5)), rand(5, 1)))

  for m in [R, L, G, G3]

    x1 = rand(3)
    x2 = rand(3, 1)
    x3 = rand(3, 1, 2)
    Flux.reset!(m)
    @test size(m(x1)) == (5,)
    Flux.reset!(m)
    @test size(m(x1)) == (5,) # repeat in case of effect from change in state shape
    @test size(m(x2)) == (5, 1)
    Flux.reset!(m)
    @test size(m(x2)) == (5, 1)
    Flux.reset!(m)
    @test size(m(x3)) == (5, 1, 2)
    Flux.reset!(m)
    @test size(m(x3)) == (5, 1, 2)
  end
end

@testset "type matching" begin
  x = rand(Float64, 2, 4)
  m1 = RNN(2=>3)
  @test m1(x) isa Matrix{Float32}  # uses _match_eltype, may print a warning
  @test m1.state isa Matrix{Float32}
  @test (@inferred m1(x); true)
  @test Flux.outputsize(m1, size(x)) == size(m1(x))

  m2 = LSTM(2=>3)
  @test m2(x) isa Matrix{Float32}
  @test (@inferred m2(x); true)
  @test Flux.outputsize(m2, size(x)) == size(m2(x))

  m3 = GRU(2=>3)
  @test m3(x) isa Matrix{Float32}
  @test (@inferred m3(x); true)
  @test Flux.outputsize(m3, size(x)) == size(m3(x))

  m4 = GRUv3(2=>3)
  @test m4(x) isa Matrix{Float32}
  @test (@inferred m4(x); true)
  @test Flux.outputsize(m4, size(x)) == size(m4(x))
end