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

const cutanh = CUDAnative.tanh

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
    h̃ = cutanh.(Cx'*x .+ r .* Ch'*h .+ bC)
    h′ = (1.-z).*h̃ .+ z.*h
    return h′, h′
  elseif rnn.mode == CUDA.LSTM
    Ix, Fx, Cx, Ox, Ih, Fh, Ch, Oh = rnn.weights
    bI, bF, bC, bO = rnn.biases
    input = σ.(Ix'*x .+ Ih'*h .+ bI)
    forget = σ.(Fx'*x .+ Fh'*h .+ bF)
    cell = cutanh.(Cx'*x .+ Ch'*h .+ bC)
    output = σ.(Ox'*x .+ Oh'*h .+ bO)
    c = forget .* c .+ input .* cell
    h = output .* cutanh.(c)
    return (h, h, c)
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

rnn = RNNDesc{Float32}(CUDA.LSTM, 10, 5)
randinit(rnn)
x, h, c = cu(rand(10)), cu(rand(5)), cu(rand(5))
@test collect(test_forward(rnn, x, h, c)[1]) ≈ collect(CUDA.forwardInference(rnn, x, h, c)[1])
@test collect(test_forward(rnn, x, h, c)[2]) ≈ collect(CUDA.forwardInference(rnn, x, h, c)[2])

end
