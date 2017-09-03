using Juno
using Flux.Tracker: back!

function train!(m, data, opt; epoch = 1, cb = () -> ())
  @progress for e in 1:epoch
    epoch > 1 && info("Epoch $e")
    @progress for (x, y) in data
      back!(m(x, y))
      opt()
      cb()
    end
  end
end
