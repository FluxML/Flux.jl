# Ref FluxML/Flux.jl#1209 1D input
@testset "BPTT-1D" begin
  seq = [rand(Float32, 2) for i = 1:3]
  for r ∈ [RNN,]
    rnn = r(2, 3)
    Flux.reset!(rnn)
    grads_seq = gradient(Flux.params(rnn)) do
        sum(rnn.(seq)[3])
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
    rnn = r(2, 3)
    Flux.reset!(rnn)
    grads_seq = gradient(Flux.params(rnn)) do
        sum(rnn.(seq)[3])
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

@testset "RNN-shapes" begin
    @testset for R in [RNN, GRU, LSTM]
        m1 = R(3, 5)
        m2 = R(3, 5)
        x1 = rand(Float32, 3)
        x2 = rand(Float32,3,1)
        Flux.reset!(m1)
        Flux.reset!(m2)
        @test size(m1(x1)) == (5,)
        @test size(m1(x1)) == (5,) # repeat in case of effect from change in state shape
        @test size(m2(x2)) == (5,1)
        @test size(m2(x2)) == (5,1)
    end
end

@testset "RNN-input-state-eltypes" begin
  @testset for R in [RNN, GRU, LSTM]
      m = R(3, 5)
      x = rand(Float64, 3, 1)
      Flux.reset!(m)
      @test_throws MethodError m(x)
  end
end