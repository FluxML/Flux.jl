using Juno
import Flux.Tracker: Params, gradient, data, update!
import Base.depwarn

function update!(opt, x, x̄)
  update!(x, -apply!(opt, x, data(x̄)))
end

function update!(opt, xs::Params, gs)
  for x in xs
    update!(opt, x, gs[x])
  end
end

# Added as an internal API but everyone started using it.
function _update_params!(opt, xs)
  depwarn("`_update_params!` is deprecated, use `update!` instead.", :stop)
  for x in xs
    update!(opt, x, Tracker.grad(x))
    x.tracker.grad = Tracker.zero_grad!(x.tracker.grad)
  end
end

# Callback niceties
call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

struct StopException <: Exception end
"""
    stop()

Call `Flux.stop()` in a callback to indicate when a callback condition is met.
This would trigger the train loop to stop and exit.

```julia
# Example callback:

cb = function ()
  accuracy() > 0.9 && Flux.stop()
end
```
"""
function stop()
  throw(StopException())
end

"""
    train!(loss, params, data, opt; cb)

For each datapoint `d` in `data` computes the gradient of `loss(d...)` through
backpropagation and calls the optimizer `opt`.

Takes a callback as keyword argument `cb`. For example, this will print "training"
every 10 seconds:

```julia
Flux.train!(loss, params, data, opt,
            cb = throttle(() -> println("training"), 10))
```

The callback can call `Flux.stop()` to interrupt the training loop.

Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function train!(loss, ps, data, opt; cb = () -> ())
  ps = Params(ps)
  cb = runall(cb)
  @progress for d in data
    try
      gs = gradient(ps) do
        loss(d...)
      end
      update!(opt, ps, gs)
      if cb() == :stop
        depwarn("Use of `:stop` is deprecated; use `Flux.stop()` instead", :stop)
        break
      end
    catch ex
      if ex isa StopException
        break
      else
        rethrow(ex)
      end
    end
  end
end

"""
    @epochs N body

Run `body` `N` times. Mainly useful for quickly doing multiple epochs of
training in a REPL.

```julia
julia> @epochs 2 println("hello")
INFO: Epoch 1
hello
INFO: Epoch 2
hello
```
"""
macro epochs(n, ex)
  :(@progress for i = 1:$(esc(n))
      @info "Epoch $i"
      $(esc(ex))
    end)
end
