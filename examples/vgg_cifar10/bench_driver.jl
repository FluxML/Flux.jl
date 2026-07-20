# Runs one config, chosen from ARGS: method batchsize epochs ntrain
include("vgg_cifar10.jl")
method    = Symbol(get(ARGS, 1, "train_step"))
batchsize = parse(Int, get(ARGS, 2, "128"))
epochs    = parse(Int, get(ARGS, 3, "3"))
ntrain    = parse(Int, get(ARGS, 4, "5120"))
main(; method, batchsize, epochs, ntrain)
