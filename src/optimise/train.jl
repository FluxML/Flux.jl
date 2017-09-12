using Juno
using Flux.Tracker: back!

tocb(f) = f
tocb(fs::AbstractVector) = () -> foreach(call, fs)

function train!(m, data, opt; cb = () -> ())
  cb = tocb(cb)
  @progress for x in data
    l = m(x...)
    isinf(l.data[]) && error("Inf")
    isnan(l.data[]) && error("NaN")
    back!(l)
    opt()
    cb()
  end
end
