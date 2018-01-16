using Juno
using Flux.Tracker: back!, value

runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

"""
    train!(loss, data, opt)

For each datapoint `d` in `data` computes the gradient of `loss(d...)` through
backpropagation and calls the optimizer `opt`.

Takes a callback as keyword argument `cb`. For example, this will print "training"
every 10 seconds:

```julia
Flux.train!(loss, data, opt,
            cb = throttle(() -> println("training"), 10))
```

The callback can return `:stop` to interrupt the training loop.

Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function train!(loss, data, opt; cb = () -> ())
  cb = runall(cb)
  opt = runall(opt)
  @progress for d in data
    l = loss(d...)
    isinf(value(l)) && error("Loss is Inf")
    isnan(value(l)) && error("Loss is NaN")
    back!(l)
    opt()
    cb() == :stop && break
  end
end
