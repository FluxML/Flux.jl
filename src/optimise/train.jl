using Juno
using Flux.Tracker: back!

tocb(f) = f
tocb(fs::AbstractVector) = () -> foreach(call, fs)

function train!(m, data, opt; cb = () -> ())
  cb = tocb(cb)
  @progress for x in data
    back!(m(x...))
    opt()
    cb()
  end
end
