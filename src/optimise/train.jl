using Juno
using Flux.Tracker: back!

tocb(f) = f
tocb(fs::AbstractVector) = () -> foreach(call, fs)

function train!(loss, data, opt; cb = () -> ())
  cb = tocb(cb)
  @progress for x in data
    l = loss(x...)
    isinf(l.data[]) && error("Loss is Inf")
    isnan(l.data[]) && error("Loss is NaN")
    back!(l)
    opt()
    cb()
  end
end
