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
    train!(loss, params, data, opt; cb)

For each datapoint `d` in `data` computes the gradient of `loss(d...)` through
backpropagation and calls the optimizer `opt`.

Takes a callback as keyword argument `cb`. For example, this will print "training"
every 10 seconds:

```julia
Flux.train!(loss, params, data, opt,
            cb = throttle(() -> println("training"), 10))
```

The callback can return `:stop` to interrupt the training loop.

Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function train!(loss, ps, data, opt; cb = () -> ())
  cb = runall(cb)
  opt = runall(opt)
  @progress for d in data
    try
      l = loss(d...)
      @interrupts back!(l)
      update!(opt, ps)
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
