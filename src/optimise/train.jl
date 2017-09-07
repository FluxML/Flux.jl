using Juno
using Flux.Tracker: back!

tocb(f) = f
tocb(fs::AbstractVector) = () -> foreach(call, fs)

function train!(m, data, opt; cb = () -> ())
  cb = tocb(cb)
  @progress for (x, y) in data
    back!(m(x, y))
    opt()
    cb()
  end
end
