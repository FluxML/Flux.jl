using Flux
using CUDA
using Flux: DataLoader
import Optimisers
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Statistics, Random
import MLDatasets


function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtest, ytest = MLDatasets.MNIST(:test)[:]

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=args.batchsize)
    
    return train_loader, test_loader
end

function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    conv_block = Chain(Conv((5, 5), imgsize[end]=>6, relu),
                        MaxPool((2, 2)),
                        Conv((5, 5), 6=>16, relu),
                        MaxPool((2, 2)))

    out_conv_size = Flux.outputsize(conv_block, imgsize, padbatch=true)
    
    head =  Chain(flatten,
                Dense(prod(out_conv_size), 120, relu), 
                Dense(120, 84, relu), 
                Dense(84, nclasses))

    return Chain(; conv_block, head)
end

function MLP(; imgsize=(28,28,1), nclasses=10) 
    return  Chain(flatten,
                Dense(prod(imgsize) => 120, relu), 
                Dense(120 => 120, relu), 
                Dense(120 => nclasses))
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)


function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]        
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

round4(x) = round(x, digits=4)

Base.@kwdef mutable struct Args
    η = 3e-4             ## learning rate
    batchsize = 128      ## batch size
    epochs = 10          ## number of epochs
    seed = 0             ## set seed > 0 for reproducibility
    use_cuda = true      ## if true use cuda (if available)
    infotime = 1 	     ## report every `infotime` epochs
end

# function train(; kws...)
    # args = Args(; kws...)
args = Args(epochs=2)
args.seed > 0 && Random.seed!(args.seed)
use_cuda = args.use_cuda && CUDA.functional()

if use_cuda
    device = gpu ∘ f16
    @info "Training on GPU"
else
    device = cpu ∘ f16
    @info "Training on CPU"
end

Flux.f16(x::Flux.OneHotArray) = x
Flux.f32(x::Flux.OneHotArray) = x


train_loader, test_loader = get_data(args)

model = MLP() |> device
# model = LeNet5() |> device
opt = Optimisers.setup(Optimisers.Adam(), model)

function report(epoch)
    train = eval_loss_accuracy(train_loader, model, device)
    test = eval_loss_accuracy(test_loader, model, device)        
    println("Epoch: $epoch   Train: $(train)   Test: $(test)")
end

@time report(0)
for epoch in 1:args.epochs
    for (x, y) in train_loader
        x, y = x |> device, y |> device
        grad = Flux.gradient(model -> loss(model(x), y), model)[1]
        Flux.update!(opt, model, grad)
    end
    epoch % args.infotime == 0 && report(epoch)
end

using BenchmarkTools
x = rand(Float16, 201, 201)
@btime sum(x * x)