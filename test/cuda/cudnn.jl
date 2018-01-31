using Flux, CuArrays, Base.Test
using Flux.CUDA
using Flux.CUDA: RNNDesc
using CUDAnative

info("Testing Flux/CUDNN")

function randinit(r::RNNDesc{T}) where T
  for w in (r.weights..., r.bias)
    copy!(w, randn(T, size(w)))
  end
end

const cutanh = CUDAnative.tanh

gate(rnn, x, n) = x[(1:rnn.hidden) + rnn.hidden*(n-1)]

function test_forward(rnn::RNNDesc, x, h, c = nothing)
  if rnn.mode == CUDA.RNN_RELU
    Wx, Wh = rnn.weights
    b = rnn.bias
    h′ = relu.(Wx'*x .+ Wh'*h .+ b)
    return h′, h′
  elseif rnn.mode == CUDA.GRU
    Wx, Wh = rnn.weights
    b = rnn.bias
    gx, gh = Wx'*x, Wh'*h
    r = σ.(gate(rnn, gx, 1) .+ gate(rnn, gh, 1) .+ gate(rnn, b, 1))
    z = σ.(gate(rnn, gx, 2) .+ gate(rnn, gh, 2) .+ gate(rnn, b, 2))
    h̃ = cutanh.(gate(rnn, gx, 3) .+ r .* gate(rnn, gh, 3) .+ gate(rnn, b, 3))
    h′ = (1.-z).*h̃ .+ z.*h
    return h′, h′
  elseif rnn.mode == CUDA.LSTM
    Wx, Wh = rnn.weights
    b = rnn.bias
    g = Wx'*x .+ Wh'*h .+ b
    input = σ.(gate(rnn, g, 1))
    forget = σ.(gate(rnn, g, 2))
    cell = cutanh.(gate(rnn, g, 3))
    output = σ.(gate(rnn, g, 4))
    c = forget .* c .+ input .* cell
    h = output .* cutanh.(c)
    return (h, h, c)
  end
end

@testset "CUDNN" begin

rnn = RNNDesc{Float32}(CUDA.RNN_RELU, 10, 5)
randinit(rnn)
x, h = cu(rand(10)), cu(rand(5))
@test collect(test_forward(rnn, x, h)[1]) ≈
  collect(CUDA.forwardInference(rnn, x, h)[1])

rnn = RNNDesc{Float32}(CUDA.GRU, 10, 5)
randinit(rnn)
x, h = cu(rand(10)), cu(rand(5))
@test collect(test_forward(rnn, x, h)[1]) ≈
  collect(CUDA.forwardInference(rnn, x, h)[1])

rnn = RNNDesc{Float32}(CUDA.LSTM, 10, 5)
randinit(rnn)
x, h, c = cu(rand(10)), cu(rand(5)), cu(rand(5))
@test collect(test_forward(rnn, x, h, c)[1]) ≈
  collect(CUDA.forwardInference(rnn, x, h, c)[1])
@test collect(test_forward(rnn, x, h, c)[2]) ≈
  collect(CUDA.forwardInference(rnn, x, h, c)[2])

end
