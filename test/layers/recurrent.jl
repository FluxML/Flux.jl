# Ref FluxML/Flux.jl#1209 1D input
@testset "BPTT-1D" begin
  seq = [rand(Float32, 2) for i = 1:3]
  for r ∈ [RNN,]
    rnn = r(2 => 3)
    Flux.reset!(rnn)
    grads_seq = gradient(Flux.params(rnn)) do
        sum([rnn(s) for s in seq][3])
    end
    Flux.reset!(rnn);
    bptt = gradient(Wh -> sum(tanh.(rnn.cell.Wi * seq[3] + Wh *
                                  tanh.(rnn.cell.Wi * seq[2] + Wh *
                                        tanh.(rnn.cell.Wi * seq[1] +
                                            Wh * rnn.cell.state0
                                        + rnn.cell.b)
                                  + rnn.cell.b)
                            + rnn.cell.b)),
                    rnn.cell.Wh)
    @test grads_seq[rnn.cell.Wh] ≈ bptt[1]
  end
end

# Ref FluxML/Flux.jl#1209 2D input
@testset "BPTT-2D" begin
  seq = [rand(Float32, (2, 1)) for i = 1:3]
  for r ∈ [RNN,]
    rnn = r(2 => 3)
    Flux.reset!(rnn)
    grads_seq = gradient(Flux.params(rnn)) do
        sum([rnn(s) for s in seq][3])
    end
    Flux.reset!(rnn);
    bptt = gradient(Wh -> sum(tanh.(rnn.cell.Wi * seq[3] + Wh *
                                  tanh.(rnn.cell.Wi * seq[2] + Wh *
                                        tanh.(rnn.cell.Wi * seq[1] +
                                            Wh * rnn.cell.state0
                                        + rnn.cell.b)
                                  + rnn.cell.b)
                            + rnn.cell.b)),
                    rnn.cell.Wh)
    @test grads_seq[rnn.cell.Wh] ≈ bptt[1]
  end
end

@testset "BPTT-3D" begin
  seq = rand(Float32, (2, 1, 3))
  rnn = RNN(2 => 3)
  Flux.reset!(rnn)
  grads_seq = gradient(Flux.params(rnn)) do
    sum(rnn(seq)[:, :, 3])
  end
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
  @test grads_seq[rnn.cell.Wh] ≈ bptt[1]
end

@testset "RNN-shapes" begin
  @testset for R in [RNN, GRU, LSTM, GRUv3]
    m1 = R(3 => 5)
    m2 = R(3 => 5)
    m3 = R(3, 5)  # leave one to test the silently deprecated "," not "=>" notation
    x1 = rand(Float32, 3)
    x2 = rand(Float32, 3, 1)
    x3 = rand(Float32, 3, 1, 2)
    Flux.reset!(m1)
    Flux.reset!(m2)
    Flux.reset!(m3)
    @test size(m1(x1)) == (5,)
    @test size(m1(x1)) == (5,) # repeat in case of effect from change in state shape
    @test size(m2(x2)) == (5, 1)
    @test size(m2(x2)) == (5, 1)
    @test size(m3(x3)) == (5, 1, 2)
    @test size(m3(x3)) == (5, 1, 2)
  end
end

@testset "RNN-input-state-eltypes" begin
  @testset for R in [RNN, GRU, LSTM, GRUv3]
    m = R(3 => 5)
    x = rand(Float64, 3, 1)
    Flux.reset!(m)
    @test_throws MethodError m(x)
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
