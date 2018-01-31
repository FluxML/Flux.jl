using Flux, CuArrays, Base.Test
using Flux.CUDA
using Flux.CUDA: RNNDesc
using CUDAnative

info("Testing Flux/CUDNN")

function randinit(r::RNNDesc{T}) where T
  for w in r.weights
    copy!(w, randn(T, size(w)))
  end
  for w in r.biases
    copy!(w, randn(T, size(w)))
  end
end

function test_forward(rnn::RNNDesc, x, h, c = nothing)
  if rnn.mode == CUDA.RNN_RELU
    Wx, Wh = rnn.weights
    b, = rnn.biases
    h′ = relu.(Wx'*x .+ Wh'*h .+ b)
    return h′, h′
  elseif rnn.mode == CUDA.GRU
    Rx, Ux, Cx, Rh, Uh, Ch = rnn.weights
    bR, bU, bC = rnn.biases
    r = σ.(Rx'*x .+ Rh'*h .+ bR)
    z = σ.(Ux'*x .+ Uh'*h .+ bU)
    h̃ = CUDAnative.tanh.(Cx'*x .+ r .* Ch'*h .+ bC)
    h′ = (1.-z).*h̃ .+ z.*h
    return h′, h′
  end
end

@testset "CUDNN" begin

rnn = RNNDesc{Float32}(CUDA.RNN_RELU, 10, 5)
randinit(rnn)
x, h = cu(rand(10)), cu(rand(5))
@test collect(test_forward(rnn, x, h)[1]) ≈ collect(CUDA.forwardInference(rnn, x, h)[1])

rnn = RNNDesc{Float32}(CUDA.GRU, 10, 5)
randinit(rnn)
x, h = cu(rand(10)), cu(rand(5))
@test collect(test_forward(rnn, x, h)[1]) ≈ collect(CUDA.forwardInference(rnn, x, h)[1])

end
