using Juno
import Flux.Tracker: data, grad, back!, update!
import Base.depwarn

function update!(opt, x, x̄)
  update!(x, apply!(opt, x, copy(data(x̄))))
end


"""
  update!(parameters, opt, losses)

Update the model `parameters` , via the optimizer `opt`,  to minimize the loss given by `losses`
When writing your own training loop, this function should be at the core.

For example, a simple training loop:
(So simply in practicve you would just use `train!`)

```julia
# Inputs: model, xs, ys, opt 
ps = params(model)
for (x,y_true) in zip(xs, ys)
    y_pred = model(x)
    loss = (y_pred - y_true)^2
    update!(ps, opt, loss)
end
```

It is useful to construct your own training loop like this when you want more control than
simply using the standard  [train!](@ref).
For example if your loss function changes depending on the iteration, or some external condition,
or you want to use a complicated early stopping rule.
While all things can be done via surficiently complicated closures in the [train!](@ref) callbacks and loss functions,
it is often cleaner to just write your own training loop, using `update!` as above.

A more complicated example:
```julia
# Inputs: model, xs, ys, opt 
ps = params(model)
num_obs = length(xs)

for (ii, (x, y_true)) in enumerate(zip(xs, ys))
    y_pred = model(x)
    loss = (y_pred - y_true)^2
    loss *= exp10(10*ii/num_obs)  # Weight losses proportionate to how recently 
    update!(ps, opt, loss)
    
    @info "traing step done" ii loss
end
```

"""
function update!(parameters::Params, opt, losses)
    @interrupts back!(losses)
    _update_params!(opt, parameters)
end

function _update_params!(opt, xs)
  for x in xs
    Δ = apply!(opt, x.data, x.grad)
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

The callback can call `Flux.stop()` to interrupt the training loop.

Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function train!(loss, ps, data, opt; cb = () -> ())
  cb = runall(cb)
  opt = runall(opt)
  @progress for d in data
    try
      l = loss(d...)
      update!(ps, opt, l)
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
