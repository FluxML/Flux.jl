using Juno
using Flux.Tracker: data, grad, back!
import Base.depwarn

function update!(opt, xs)
  for x in xs
    Δ = update!(opt, x.data, x.grad)
    x.data .-= Δ
    Δ .= 0
  end
end

# Callback niceties
call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

# The AD generates fairly large backtraces that are unhelpful if you interrupt
# while training; this just cleans that up.
macro interrupts(ex)
  :(try $(esc(ex))
    catch e
      e isa InterruptException || rethrow()
      throw(e)
    end)
end

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
    CallbackInfo(
        datapoint,
        datapoins_count,
        iteration_count,
        loss
    )

Info about the previous training pass in Flux.train!

`datapoint` - model input in the previous training pass.

`datapoints_count` - number of datapoints in the dataset.

`iteration_count` - counter that increases with each training pass.

`loss` - value returned by the loss function in the previous training pass.
"""
struct CallbackInfo
  datapoint
  datapoint_count
  iteration_count
  loss
end

"""
    train!(loss, params, data, opt)

For each datapoint `d` in `data` computes the gradient of `loss(d...)` through
backpropagation and calls the optimizer `opt`.

Takes a callback as keyword argument `cb`. For example, this will print "training"
every 10 seconds:

```julia
Flux.train!(loss, params, data, opt,
            cb = throttle((_) -> println("training"), 10))
```

The callback can take a CallbackInfo value as a single argument that provides info
on the previous training pass. For example, this will print the training loss with
its iteration count:

```julia
Flux.train!(loss, params, data, opt,
            cb = (info) -> println("Iteration \$(info.iteration_count): Loss \$loss."))
```
See `?Flux.CallbackInfo` for more details.

The callback can return `:stop` to interrupt the training loop.

Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function train!(loss, ps, data, opt; cb = () -> ())
  opt = runall(opt)
  @progress for (index, d) in enumerate(data)
    try
      l = loss(d...)
      @interrupts back!(l)
      update!(opt, ps)
      cbinfo = CallbackInfo(d, length(data), index, l)
      add_cbinfo_if_wanted(cb) = methods(cb).ms[1].nargs == 1 ? cb : () -> cb(cbinfo)
      runcb = !(typeof(cb) <: AbstractVector) ?
        runall(add_cbinfo_if_wanted(cb)) :
        runall(map(add_cbinfo_if_wanted, cb))
      if runcb() == :stop
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
