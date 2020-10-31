using Revise
using Flux
# using Zygote
# using CUDA
using Random
using Statistics: mean

################################################
# Define operators
################################################
struct MyRecur{T}
    cell::T
    state
end

function (m::MyRecur)(x)
    y = m.cell(m.state, x)
    # m.state = h
    return y
end

# Flux.@functor MyRecur cell, init
# Flux.@functor MyRecur (cell,)
Flux.@functor MyRecur
Flux.trainable(a::MyRecur) = (a.cell,)
# Flux.trainable(a::MyRecur) = (a.cell,)

function reset!(m::MyRecur) 
    m.state = m.cell.init
    # m.state .*= 0.0
end
reset!(m) = foreach(reset!, Flux.functor(m)[1])

# Vanilla RNN
mutable struct MyRNNCell{F,A,V}
    σ::F
    Wi::A
    Wh::A
    b::V
    init::V
end

MyRNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform) = MyRNNCell(σ, init(out, in), init(out, out), init(out), zeros(Float32, out))

function (m::MyRNNCell)(h, x)
    σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
    h = Wi * x .+ Wh * h .+ b
    # h = σ.(Wi * x .+ Wh * h .+ b)
    # return h, h
    return h
end

# init(m::MyRNNCell) = m.init
Flux.@functor MyRNNCell
Flux.trainable(a::MyRNNCell) = (a.Wi, a.Wh, a.b)
MyRecur(m::MyRNNCell) = MyRecur(m, Flux.glorot_uniform(length(m.init)))
MyRNN(a...; ka...) = MyRecur(MyRNNCell(a...; ka...))


# RNN test gpu
function debug_rnn(seed=123)

    # illustrate diverging behavior of GPU execution
    feat = 2
    h_size = 3
    seq_len = 1
    batch_size = 1

    function loss(x, y)
        # reset!(rnn)
        # l1 = (Flux.stack(map(rnn, x), 2) .- y)
        # println("l1: ", l1)
        l = mean((Flux.stack(map(rnn, x), 2) .- y).^2)
        # l = mean((Flux.stack(map(rnn, x), 2) .- y))
        # l = mean((rnn(x[1]) .- y).^2f0)
        # l = mean(rnn(x[1]))
        # reset!(rnn)
        return l
    end
    function loss_gpu(x, y)
        # reset!(rnn_gpu)
        # l1 = (Flux.stack(map(rnn_gpu, x), 2) .- y)
        # println("l1 gpu: ", l1)
        l = mean((Flux.stack(map(rnn_gpu, x), 2) .- y).^2)
        # l = mean((Flux.stack(map(rnn_gpu, x), 2) .- y))
        # l = mean(rnn_gpu(x[1] .- y))
        # reset!(rnn_gpu)
        return l
    end

    Random.seed!(seed)
    rnn = Chain(MyRNN(feat, h_size),
        Dense(h_size, 1),
        x -> reshape(x, :))

    X = [rand(Float32, feat, batch_size) for i in 1:seq_len]
    Y = rand(Float32, batch_size, seq_len) ./ 10

    # transfer to gpu ####
    rnn_gpu = rnn |> gpu
    X_gpu = gpu(X)
    Y_gpu = gpu(Y)

    θ = Flux.params(rnn)
    θ_gpu = Flux.params(rnn_gpu)

    opt = Descent(1e-2)
    opt_gpu = Descent(1e-2)

    l_cpu = loss(X,Y)
    l_gpu = loss_gpu(X_gpu,Y_gpu)

    # println("loss_ratio: ", l_cpu / l_gpu - 1)

    println("CPU")
    println("loss: ", loss(X,Y))
    # println("X: ", X)
    # println("Y: ", Y)
    debug_train!(loss, θ, (X,Y))
    println()

    println("GPU")
    println("loss: ", loss_gpu(X_gpu,Y_gpu))
    # println("X_gpu: ", X_gpu)
    # println("Y_gpu: ", Y_gpu)
    debug_train!(loss_gpu, θ_gpu, (X_gpu,Y_gpu))
    println()
end

function debug_train!(loss, ps, d)
    gs = gradient(ps) do
        loss(d...)
    end
    # x = first(ps)
    @show ps[1] gs[ps[1]]
    @show ps[2] gs[ps[2]]
    @show ps[3] gs[ps[3]]
    @show ps[4] gs[ps[4]]
    @show ps[5] gs[ps[5]]
end

# seed 19 and 77 are buggy
debug_rnn(15)
debug_rnn(19)
debug_rnn(77)

for i in 101:200
    println(i)
    debug_rnn(i)
end

