using Flux, FiniteDifferences, Test, Zygote, Functors, Metal
include("test/test_utils.jl")

m = Dense(3, 3)
x = rand(Float32, 3, 3)
test_gradients(m, x; rtol=1e-4, atol=1e-4)

m = MultiHeadAttention(4, nheads=2)
x = rand(Float32, 4, 3, 2)
m(x)
test_gradients(m, x; loss = o -> sum(o[1].^2) + sum(o[2].^2))
