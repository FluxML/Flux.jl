using Flux
using Base.Test

module MNISTBasic
include("../examples/MNIST.jl")
end

module MNISTConv
include("../examples/mnist-conv.jl")
end
