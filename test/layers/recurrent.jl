using Flux, Test
using Flux: gradient, Optimise, reset!, learn_initial_state!
using Flux.Optimise: update!

dummyloss(rnn, x) = sum(rnn(x))

function test_init_reset_nolearn(rnn, x, test_init)
  opt = ADAM()
  ps = params(rnn)
  @test rnn.init == test_init
  @test rnn.state == test_init
  gs = gradient(()->dummyloss(rnn, x), ps)
  update!(opt, ps, gs)
  @test rnn.state != test_init
  @test rnn.init == test_init
  reset!(rnn)
  @test rnn.state == test_init
end

@testset "RNN init/reset default workflow" begin
  test_init_reset_nolearn(RNN(5, 10), rand(5), zeros(10))
  test_init_reset_nolearn(LSTM(5, 10), rand(5), (zeros(10), zeros(10)))
  test_init_reset_nolearn(GRU(5, 10), rand(5), zeros(10))
end
