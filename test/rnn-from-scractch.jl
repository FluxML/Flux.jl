using Revise
using Flux
using Zygote
using CUDA
using Statistics: mean

################################################
# Define operators
################################################
mutable struct MyRecur{T}
    cell::T
    init
    state
end

MyRecur(m, h=hidden(m)) = MyRecur(m, h, h)

function (m::MyRecur)(xs...)
    h, y = m.cell(m.state, xs...)
    m.state = h
    return y
end

# Flux.@functor MyRecur cell, init
Flux.@functor MyRecur
Flux.trainable(a::MyRecur) = (a.cell, a.init)
# Flux.trainable(a::MyRecur) = (a.cell,)

reset!(m::MyRecur) = (m.state = m.init)
reset!(m) = foreach(reset!, functor(m)[1])

# Vanilla RNN
struct MyRNNCell{F,A,V}
    σ::F
    Wi::A
    Wh::A
    b::V
end

MyRNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform) =
MyRNNCell(σ, init(out, in), init(out, out), init(out))

function (m::MyRNNCell)(h, x)
    σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
    h = σ.(Wi * x .+ Wh * h .+ b)
    return h, h
end

hidden(m::MyRNNCell) = m.h
Flux.@functor MyRNNCell

MyRecur(m::MyRNNCell) = MyRecur(m, zeros(Float32, length(m.b)), zeros(Float32, length(m.b)))
MyRNN(a...; ka...) = MyRecur(MyRNNCell(a...; ka...))

########################
# RNN test gpu
########################

# illustrate diverging behavior of GPU execution
feat = 32
h_size = 64
seq_len = 20
batch_size = 100

rnn = Chain(MyRNN(feat, h_size),
  Dense(h_size, 1, σ),
  x -> reshape(x, :))

X = [rand(Float32, feat, batch_size) for i in 1:seq_len]
Y = rand(Float32, batch_size, seq_len) ./ 10

#### transfer to gpu ####
rnn_gpu = rnn |> gpu
X_gpu = gpu(X)
Y_gpu = gpu(Y)

θ = Flux.params(rnn)
θ_gpu = Flux.params(rnn_gpu)
function loss(x, y)
    l = mean((Flux.stack(map(rnn, x), 2) .- y).^2f0)
    Flux.reset!(rnn)
    return l
end
function loss_gpu(x, y)
    l = mean((Flux.stack(map(rnn_gpu, x), 2) .- y).^2f0)
    Flux.reset!(rnn_gpu)
    return l
end

opt = Descent(1e-2)
opt_gpu = Descent(1e-2)
for i in 1:25
    println("iter: ", i)
    Flux.train!(loss, θ, [(X, Y)], opt)
    Flux.train!(loss_gpu, θ_gpu, [(X_gpu, Y_gpu)], opt_gpu)
    println("loss_cpu: ", loss(X, Y))
    println("loss_gpu: ", loss_gpu(X_gpu, Y_gpu))
    println("θ[3][1:2]: ", θ[3][1:2])
    println("θ_gpu[3][1:2]: ", θ_gpu[3][1:2])
    println("θ[4][1:2]: ", θ[4][1:2])
    println("θ_gpu[4][1:2]: ", θ_gpu[4][1:2])
    println("rnn.layers[1].state[1:2]: ", rnn.layers[1].state[1:2])
    println("rnn_gpu.layers[1].state[1:2]: ", rnn_gpu.layers[1].state[1:2])
end
# loss should show an NA if run on GPU, but not on CPU after going over 100 iterations
typeof(loss(X, Y))
typeof(loss_gpu(X_gpu, Y_gpu))

loss(X,Y)
loss_gpu(X_gpu, Y_gpu)
Flux.reset!(rnn)
Flux.train!(loss, θ, [(X, Y)], opt)

# @code_warntype loss(X, Y)
# @code_warntype loss_gpu(X_gpu, Y_gpu)

rnn.layers[1].init
rnn.layers[1].state
rnn_gpu.layers[1].init
rnn_gpu.layers[1].state

θ[1]
θ[2]
θ[3]
θ[4]
θ[5]
θ[6]

θ_gpu[4] .= 0

rnn_gpu(X_gpu[1])
