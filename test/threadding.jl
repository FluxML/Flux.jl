using Revise
using Flux
using Base.Threads
using Iterators

makedataset() = (hcat(randn(4,50),randn(4,50).+3),vcat(ones(50),2ones(50)))

model = Chain(Dense(4,2),softmax)
loss(f,ds) = Flux.crossentropy(f(ds[1]),Flux.onehotbatch(ds[2],1:2))

data = Iterators.repeated([makedataset() for i in 1:nthreads()],100)
opt = Flux.Optimise.ADAM(params(model))
Flux.train_threaded(model,loss,data,opt;cb = () -> ())