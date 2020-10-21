using Revise
using Flux
# using CUDA
using Statistics: mean

########################
# RNN test gpu
########################

# illustrate diverging behavior of GPU execution
feat = 16
h_size = 32
seq_len = 10
batch_size = 100

rnn = Chain(RNN(feat, h_size),
  Dense(h_size, 1, σ),
  x -> reshape(x,:))

X = [rand(feat, batch_size) for i in 1:seq_len]
Y = rand(batch_size, seq_len) ./ 10

######################################
#### do not run for cpu execution ####
rnn_gpu = rnn |> gpu
X_gpu = gpu(X)
Y_gpu = gpu(Y)
######################################

θ = Flux.params(rnn)
θ_gpu = Flux.params(rnn_gpu)
function loss(x,y)
  l = mean((Flux.stack(map(rnn, x),2) .- y) .^ 2f0)
  Flux.reset!(rnn)
  return l
end
function loss_gpu(x,y)
  l = mean((Flux.stack(map(rnn_gpu, x),2) .- y) .^ 2f0)
  Flux.reset!(rnn_gpu)
  return l
end

opt = ADAM(1e-4)
opt_gpu = ADAM(1e-4)
# loss(X,Y)
# loss_gpu(X_gpu, Y_gpu)
# Flux.train!(loss, θ, [(X,Y)], opt)
# Flux.train!(loss_gpu, θ_gpu, [(X_gpu,Y_gpu)], opt_gpu)
# loss(X,Y)
# loss_gpu(X_gpu, Y_gpu)
for i in 1:50
  println("iter: ", i)
  Flux.train!(loss, θ, [(X,Y)], opt)
  Flux.train!(loss_gpu, θ_gpu, [(X_gpu,Y_gpu)], opt_gpu)
  println("loss_cpu: ", loss(X, Y))
  println("loss_gpu: ", loss_gpu(X_gpu, Y_gpu))
end
# loss should show an NA if run on GPU, but not on CPU after going over 100 iterations
loss(X,Y)
loss_gpu(X_gpu, Y_gpu)
Flux.reset!(rnn)
Flux.train!(loss, θ, [(X,Y)], opt)

rnn.layers[1].init
rnn.layers[1].state
rnn_gpu.layers[1].init
rnn_gpu.layers[1].state

θ[1]
θ[2]
θ[3]
θ[4]

######################
# basic test
######################
seq = [[1,2] ./ 10 for b in 1:3]
seq = hcat(seq...)
seq = [seq for i = 1:4]

m = RNN(2, 5)
m.cell.Wi .= [0.1 0]
m.cell.Wh .= [0.2]
m.cell.b .= 1.1
# m.cell.h .= 1.2
m.init .= 1.3 # init value stays at init value - rather than taking state value
m.state .= 1.4

params(m)
params(m)[1]
m(seq[2])
@time m.(seq)
@time map(m, seq)

######################
# single array
######################
seq = [[1,2] ./ 10 for b in 1:3]
seq = hcat(seq...)
seq = [seq for i = 1:4]
seq = cat(seq..., dims=3)

m = RNN(2, 5)
m.cell.Wi .= [0.1 0]
m.cell.Wh .= [0.1]
m.cell.b .= 0
# m.cell.h .= 0
m.init .= 0.0
m.state .= 0

params(m)
@time mapslices(m, seq, dims=(1,2))
mapslices(size, seq, dims=(1,2))



######################
# issue: https://github.com/FluxML/Flux.jl/issues/1114
######################
rnn = Chain(LSTM(16, 8),
  Dense(8,1, σ),
  x -> reshape(x,:))

X = [rand(16,10) for i in 1:20]
Y = rand(10,20) ./ 10

rnn = rnn |> gpu
X = gpu(X)
Y = gpu(Y)

θ = Flux.params(rnn)
loss(x,y) = mean((Flux.stack(rnn.(x),2) .- y) .^ 2f0)
opt = ADAM(1e-3)
size(rnn[1].state[1])
Flux.reset!(rnn)
size(rnn[1].state[1])
Flux.train!(loss, θ, [(X,Y)], opt)
size(rnn[1].state[1])
loss(X,Y)

Flux.stack(rnn.(X),2)
rnn.(X)

########################
# rnn test gpu
########################
feat = 2
h_size = 11
seq_len = 4
batch_size = 3
rnn = Chain(RNN(feat, h_size),
  Dense(h_size, 1, σ),
  x -> reshape(x,:))

X = [rand(feat, batch_size) for i in 1:seq_len]
Y = rand(batch_size, seq_len) ./ 10

rnn = rnn |> gpu
X = gpu(X)
Y = gpu(Y)

θ = Flux.params(rnn)
mapreduce(length, +, θ) - h_size -1 # num params in RNN

function loss(x,y)
  l = mean((Flux.stack(map(rnn, x),2) .- y) .^ 2f0)
  Flux.reset!(rnn)
  return l
end

opt = ADAM(1e-3)
loss(X,Y)
Flux.reset!(rnn)
Flux.train!(loss, θ, [(X,Y)], opt)
loss(X,Y)
for i in 1:100
  Flux.train!(loss, θ, [(X,Y)], opt)
end
Flux.reset!(rnn)
Flux.train!(loss, θ, [(X,Y)], opt)

θ[1]
θ[3]
θ[4]
