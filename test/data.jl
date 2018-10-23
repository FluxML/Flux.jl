using Flux.Data
using Test

@test cmudict()["CATASTROPHE"] == :[K,AH0,T,AE1,S,T,R,AH0,F,IY0].args

@test length(CMUDict.phones()) == 39

@test length(CMUDict.symbols()) == 84

@test MNIST.images()[1] isa Matrix
@test MNIST.labels() isa Vector{Int64}

@test FashionMNIST.images()[1] isa Matrix
@test FashionMNIST.labels() isa Vector{Int64}

@test Data.Sentiment.train() isa Vector{Data.Tree{Any}}
