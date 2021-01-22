using Revise
using Flux
using Flux: @functor
import Flux: trainable
using Statistics: mean
using Random: seed!


mutable struct Recur2{T,S}
    cell::T
    state::S
end

# original definition
# function (m::Recur2)(xs...)
#     m.state, y = m.cell(m.state, xs...)
#     return y
# end

# new def
function (m::Recur2)(xs...)
    m.state, y = m.cell(m.state, xs...)
    return y
end

@functor Recur2
trainable(a::Recur2) = (a.cell,)

#####################################
# Basic test
#####################################
seed!(123)
feat = 3
h_size = 5
seq_len = 7
batch_size = 4

X = [rand(Float32, feat, batch_size) for i in 1:seq_len]
Y = rand(Float32, batch_size, seq_len) ./ 10

cell = Flux.RNNCell(feat, h_size)
rnn = Recur2(cell, cell.state0)

rnn(X[1])
rnn.state
rnn(X[1])

rnn.(X)

function fold_test_1(x, m)
    foldl((a, b) -> m(b), x)
end
fold_test_1(X, rnn)

rnn.(X)

function rnn2(x)
    # println((x))
    println("state: ", rnn.state)
    rnn(x)
end
function fold_test_2(x)
    foldl((a, b) -> rnn(b), x, init=x[1])
end
fold_test_2(X)
rnn.state

function fold_cell_1(x, c)
    foldl((a, b) -> cell(a, b)[1], x, init=cell.state0)
end
fold_cell_1(X, cell)
rnn.state


f1(x) = begin
    println(x)
    x^2
end

function fold_test_2(x)
    foldl((a, b) -> f1(b), x, init=5)
end
x1 = fold_test_2([2,3])

# rnn = Chain(
#     RNN(feat, h_size),
#     Dense(h_size, 1, σ),
#     x -> reshape(x, :))


#### transfer to gpu ####
rnn_gpu = rnn |> gpu
X_gpu = gpu(X)
Y_gpu = gpu(Y)

θ = Flux.params(rnn)
θ_gpu = Flux.params(rnn_gpu)
length(θ)
length(θ_gpu)
function loss(x, y)
    Flux.reset!(rnn)
    l = mean((Flux.stack(map(rnn, x), 2) .- y).^2)
    return l
end
function loss_gpu(x, y)
    Flux.reset!(rnn_gpu)
    l = mean((Flux.stack(map(rnn_gpu, x), 2) .- y).^2)
    return l
end

opt = ADAM(1e-3)
opt_gpu = ADAM(1e-3)
for i in 1:5
    println("iter: ", i)
    Flux.train!(loss, θ, [(X, Y)], opt)
    Flux.train!(loss_gpu, θ_gpu, [(X_gpu, Y_gpu)], opt_gpu)
    println("loss_cpu: ", loss(X, Y))
    println("loss_gpu: ", loss_gpu(X_gpu, Y_gpu))
    # println("θ[3][1:2]: ", θ[3][1:2])
    # println("θ_gpu[3][1:2]: ", θ_gpu[3][1:2])
    # println("θ[4][1:2]: ", θ[4][1:2])
    # println("θ_gpu[4][1:2]: ", θ_gpu[4][1:2])
    # println("rnn.layers[1].state[1:2]: ", rnn.layers[1].state[1:2])
    # println("rnn_gpu.layers[1].state[1:2]: ", rnn_gpu.layers[1].state[1:2])
end

@code_warntype rnn(X[1])

function speed_cpu(n=10)
    for i in 1:n
        Flux.train!(loss, θ, [(X, Y)], opt)
    end
    return loss(X, Y)
end

function speed_gpu(n=10)
    for i in 1:n
        Flux.train!(loss_gpu, θ_gpu, [(X_gpu, Y_gpu)], opt_gpu)
    end
    return loss_gpu(X_gpu, Y_gpu)
end

@time speed_cpu(100)
@time speed_gpu(100)


#####################################
# RNN vanilla
#####################################
seed!(123)
feat = 32
h_size = 64
seq_len = 50
batch_size = 256

rnn = Chain(
    RNN(feat, h_size),
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
length(θ)
length(θ_gpu)
function loss(x, y)
    Flux.reset!(rnn)
    l = mean((Flux.stack(map(rnn, x), 2) .- y).^2)
    return l
end
function loss_gpu(x, y)
    Flux.reset!(rnn_gpu)
    l = mean((Flux.stack(map(rnn_gpu, x), 2) .- y).^2)
    return l
end

opt = ADAM(1e-3)
opt_gpu = ADAM(1e-3)
for i in 1:5
    println("iter: ", i)
    Flux.train!(loss, θ, [(X, Y)], opt)
    Flux.train!(loss_gpu, θ_gpu, [(X_gpu, Y_gpu)], opt_gpu)
    println("loss_cpu: ", loss(X, Y))
    println("loss_gpu: ", loss_gpu(X_gpu, Y_gpu))
    # println("θ[3][1:2]: ", θ[3][1:2])
    # println("θ_gpu[3][1:2]: ", θ_gpu[3][1:2])
    # println("θ[4][1:2]: ", θ[4][1:2])
    # println("θ_gpu[4][1:2]: ", θ_gpu[4][1:2])
    # println("rnn.layers[1].state[1:2]: ", rnn.layers[1].state[1:2])
    # println("rnn_gpu.layers[1].state[1:2]: ", rnn_gpu.layers[1].state[1:2])
end

@code_warntype rnn(X[1])

function speed_cpu(n=10)
    for i in 1:n
        Flux.train!(loss, θ, [(X, Y)], opt)
    end
    return loss(X, Y)
end

function speed_gpu(n=10)
    for i in 1:n
        Flux.train!(loss_gpu, θ_gpu, [(X_gpu, Y_gpu)], opt_gpu)
    end
    return loss_gpu(X_gpu, Y_gpu)
end

@time speed_cpu(100)
@time speed_gpu(100)

#####################################
# LSTM
#####################################
feat = 32
h_size = 64
seq_len = 50
batch_size = 256

rnn = Chain(LSTM(feat, h_size),
    LSTM(h_size, h_size),
    LSTM(h_size, h_size),
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
    Flux.reset!(rnn)
    l = mean((Flux.stack(map(rnn, x), 2) .- y).^2)
    return l
end
function loss_gpu(x, y)
    Flux.reset!(rnn_gpu)
    l = mean((Flux.stack(map(rnn_gpu, x), 2) .- y).^2)
    return l
end

opt = ADAM(1e-3)
opt_gpu = ADAM(1e-3)

for i in 1:5
    println("iter: ", i)
    Flux.train!(loss, θ, [(X, Y)], opt)
    Flux.train!(loss_gpu, θ_gpu, [(X_gpu, Y_gpu)], opt_gpu)
    println("loss_cpu: ", loss(X, Y))
    println("loss_gpu: ", loss_gpu(X_gpu, Y_gpu))
end


function speed_cpu(n=10)
    for i in 1:n
        Flux.train!(loss, θ, [(X, Y)], opt)
    end
    return loss(X, Y)
end

function speed_gpu(n=10)
    for i in 1:n
        Flux.train!(loss_gpu, θ_gpu, [(X_gpu, Y_gpu)], opt_gpu)
    end
    return loss_gpu(X_gpu, Y_gpu)
end

@code_warntype rnn(X[1])

using BenchmarkTools
@time speed_cpu(100)
@btime speed_gpu(100)

