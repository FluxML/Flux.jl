# Enzyme gradients on CuArrays.
# Compiling Enzyme for GPU code takes minutes and it does not handle every layer yet
# (BatchNorm, GroupNorm, Embedding and MultiHeadAttention error, LSTM and GRU give wrong gradients,
# LayerNorm does not finish compiling), so only a few layers are checked here.
# ConvTranspose, Bilinear, RNNCell and GRUCell pass too, but are left out to save CI time.
# Zygote is tested on GPU in the other files.

@testset "Enzyme GPU gradients" begin
  kws = (; test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=AutoEnzyme())

  @testset "Dense" begin
    test_gradients(Dense(3 => 2, tanh), rand(Float32, 3, 4); kws...)
  end

  @testset "Conv" begin
    test_gradients(Conv((2, 2), 1 => 3, tanh), rand(Float32, 28, 28, 1, 1); kws...)
  end

  @testset "LSTMCell" begin
    cell_loss(cell, x, state) = mean(cell(x, state)[1])
    state = (zeros(Float32, 3), zeros(Float32, 3))
    test_gradients(LSTMCell(2 => 3), randn(Float32, 2, 5), state; kws..., loss=cell_loss)
  end
end
