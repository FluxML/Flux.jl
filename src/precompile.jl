module Precompile

using ..Flux

function readme(gpu = identity)
  x = hcat(digits.(0:3, base=2, pad=2)...) |> gpu  # let's solve the XOR problem!
  y = Flux.onehotbatch(xor.(eachrow(cpu(x))...), 0:1) |> gpu
  data = ((Float32.(x), y) for _ in 1:100)  # an iterator making Tuples

  model = Chain(Dense(2 => 3, sigmoid), BatchNorm(3), Dense(3 => 2)) |> gpu
  optim = Adam(0.1, (0.7, 0.95))
  mloss(x, y) = Flux.logitcrossentropy(model(x), y)  # closes over model

  Flux.train!(mloss, Flux.params(model), data, optim)  # updates model & optim

  all((softmax(model(x)) .> 0.5) .== y)  # usually 100% accuracy.
end

function LeNet5(; imgsize=(28,28,1), nclasses=10)
  @autosize (imgsize..., 1) Chain(
    Conv((5, 5), _ => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), _ => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(_ => 120, relu), 
    Dense(120 => 84, relu), 
    Dense(84 => nclasses)
  )
end

function conv(gpu = identity)
  x = rand(Float32, 28, 28, 1, 100) |> gpu
  y = Flux.onehotbatch(rand(0:9, 100), 0:9) |> gpu
  data = Flux.DataLoader((x, y), batchsize=10)
  
  model = LeNet5() |> gpu
  optim = AdamW()
  mloss(x, y) = Flux.logitcrossentropy(model(x), y)
  before = mloss(x, y)
  
  Flux.train!(mloss, Flux.params(model), data, optim)
  
  after = mloss(x, y)
  before > after
end

function pretty(gpu = identity)
  io = IOBuffer()
  model = LeNet5() |> gpu
  show(io, MIME"text/plain"(), model)
  String(take!(io))
end
  
end # module