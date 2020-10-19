using Revise
using Flux
# using CUDA
using Statistics: mean

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

using CUDA

x1 = LSTM(16,8)
CUDA.CUDNN.RNNDesc(x1)
