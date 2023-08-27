using Flux
using Metal

device = Flux.get_device("Metal")
data = [(rand(Float32, 1024), rand(Float32.(0:1), 1)) for _=1:10] |> device
m = Dense(1024, 1) |> device 

loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)
opt = Flux.Train.setup(Flux.Optimise.Nesterov(), m)  # plain `Descent` works fine

Flux.Optimise.train!(loss, m, data, opt)