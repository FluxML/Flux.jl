using Flux.Tracker: back!

function train!(m, data, opt; epoch = 1)
  for e in 1:epoch
    epoch > 1 && info("Epoch $e")
    for (x, y) in data
      loss = m(x, y)
      @show loss
      back!(loss)
      update!(opt)
    end
  end
end
