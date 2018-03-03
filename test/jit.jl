using Flux, Base.Test
using Flux.JIT: compile

@testset "JIT" begin

m = Dense(10, 5)
f = compile(m, rand(10))
x = rand(10)

@test m(x) == f(x)

end
