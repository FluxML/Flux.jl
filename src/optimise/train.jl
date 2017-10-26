using Juno
using Flux.Tracker: back!

tocb(f) = f
tocb(fs::AbstractVector) = () -> foreach(call, fs)

"""
    train!(loss, data, opt; cb = () -> ())

For each datapoint `d` in `data` computes the gradient of `loss(d...)` through
backpropagation and calls the optimizer `opt` and the callback `cb`
(i.e. `opt()` and `cb()`).

Multiple callbacks can be passed to `cb` as an array.
"""
function train!(loss, data, opt; cb = () -> ())
  cb = tocb(cb)
  @progress for d in data
    l = loss(d...)
    isinf(l.data[]) && error("Loss is Inf")
    isnan(l.data[]) && error("Loss is NaN")
    back!(l)
    opt()
    cb()
  end
end
