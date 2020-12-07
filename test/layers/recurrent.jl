# Ref FluxML/Flux.jl#1209 1D input
@testset "BPTT" begin
  seq = [rand(Float32, 2) for i = 1:3]
  for r ∈ [RNN,]
    rnn = r(2,3)
    Flux.reset!(rnn)
    grads_seq = gradient(Flux.params(rnn)) do
      sum(rnn.(seq)[3])
    end
    Flux.reset!(rnn);
    bptt = gradient(Wh->sum(tanh.(rnn.cell.Wi * seq[3] + Wh *
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
@testset "BPTT" begin
  seq = [rand(Float32, (2,1)) for i = 1:3]
  for r ∈ [RNN,]
    rnn = r(2,3)
    Flux.reset!(rnn)
    grads_seq = gradient(Flux.params(rnn)) do
      sum(rnn.(seq)[3])
    end
    Flux.reset!(rnn);
    bptt = gradient(Wh->sum(tanh.(rnn.cell.Wi * seq[3] + Wh *
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