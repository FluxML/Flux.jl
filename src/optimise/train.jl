using Juno
using Flux.Tracker: back!

tocb(f) = f
tocb(fs::AbstractVector) = () -> foreach(call, fs)

function train!(m, data, opt; epoch = 1, cb = () -> ())
  cb = tocb(cb)
  @progress for e in 1:epoch
    epoch > 1 && info("Epoch $e")
    @progress for (x, y) in data
      back!(m(x, y))
      opt()
      cb()
    end
  end
end
