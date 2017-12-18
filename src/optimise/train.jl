using Juno
using Flux.Tracker: back!, value

runall(f) = f
runall(fs::AbstractVector) = (args...) -> map(f->f(args...), fs)

"""
    train!(loss, data, opt)

For each datapoint `d` in `data` computes the gradient of `loss(d...)` through
backpropagation and calls the optimizer `opt`.

Takes a callback as keyword argument `log_cb`: this is for use for logging, debugging etc.
Also optionally another callback `stopping_criteria`,
If `stopping_crieria` returns true, then the training is stopped early.
Otherwise, training will run for every datapoint.

Both callbacks takes as their arguments the step_number, and the current loss.
They are both called _before_ the network parameters are updated.

For example, the following will print the current loss every 10 steps
and will at every step check to see if the loss is less than 0.1, and if so will stop early.

```julia
Flux.train!(loss, data, opt,
            log_cb = throttle((i, l) -> println("At iter \$i, training loss=\$l"), 10),
            stopping_criteria = (i, l) -> l < 0.1
            )
```

Multiple optimisers and callbacks can be passed to `opt`, `log_cb`, and `stopping_criteria` as vectors.
"""
function train!(loss, data, opt; log_cb=(i,l) -> (), stopping_criteria=(i,l) -> false)
  log_cb = runall(log_cb)
  stopping_criteria = runall(stopping_criteria)
  opt = runall(opt)
  @progress for (step, d) in enumerate(data)
    l = loss(d...)
    isinf(value(l)) && error("Loss is Inf")
    isnan(value(l)) && error("Loss is NaN")

    log_cb(step, value(l))
    any(stopping_criteria(step, value(l))) && break

    back!(l)
    opt()
  end
end
