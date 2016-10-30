using Flux
using Base.Test

module MNISTBasic
include("../examples/MNIST.jl")
end

module MNISTConv
include("../examples/integration.jl")
end
