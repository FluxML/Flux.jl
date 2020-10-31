using Revise
using Flux
using Statistics: mean

# illustrate diverging behavior of GPU execution
feat = 32
h_size = 64
seq_len = 20
batch_size = 100

rnn = Chain(RNN(feat, h_size),
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

opt = ADAM(1e-3)
opt_gpu = ADAM(1e-3)
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

θ[1]
θ[2]
θ[3]
θ[4]
θ[5]
θ[6]



###################################
# speed test
###################################

# illustrate diverging behavior of GPU execution
feat = 32
h_size = 64
seq_len = 50
batch_size = 256

rnn = Chain(RNN(feat, h_size),
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

@time speed_cpu(100)
@time speed_gpu(100)


#####################################
# GRU
#####################################
feat = 32
h_size = 64
seq_len = 50
batch_size = 256

rnn = Chain(GRU(feat, h_size),
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

@time speed_cpu(100)
@time speed_gpu(100)