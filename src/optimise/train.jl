using Juno
using Flux.Tracker: back!

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
    isinf(l) && error("Loss is Inf")
    isnan(l) && error("Loss is NaN")
    @interrupts back!(l)
    opt()
    cb() == :stop && break
  end
end


############
# Stopping criteria

"""
    stopif(cbs...)

Given one callback, returns a callback that
returns`:stop` if that callback returns `true` or `:stop`.
i.e. converts a boolean callback, into a stopping condition.
If multiple callbacks are passed, this does the logical union of them
"""
stopif(cbs...) = () -> any(cb() ∈ (true,:stop) for cb in cbs) && :stop

"""
    atol(f, minval=0.0)

For `f` a function (e.g. a loss function, an accurasy function) taking zero arguments.
`atol` returns a callback, that returns `:stop` if `f` returns `minval` or less.
"""
atol(f, minval=0.0) = () -> f()<=minval && :stop

"""
    rtol(f, mindecrease=0.0)

For `f` a function (e.g. a loss function, an accurasy function etc) taking zero arguments.
`atol` returns a callback, that returns `:stop`
if between successive calls to the callback `f()` has not decreased by at least `mindecrease`
"""
function rtol(f, mindecrease=0.0)
    prev_score = f() # Initial score at declaration time
    function ()
        score = f()
        should_stop = prev_score - score < mindecrease
        prev_score = score
        should_stop && :stop
    end
end

"""
    max_calls(ncalls)
Returns a callback that returns :stop after it has been called `ncalls` times.
"""
function max_calls(ncalls)
    calls_so_far = 0
    function()
        calls_so_far += 1
        calls_so_far >= ncalls && :stop
    end
end


"""
    timeout(duration)
Returns a callback that returns :stop after the given duration after it was first called.
"""
function timeout(duration::Dates.TimePeriod)
    timer_started = false
    local start_time
    function()
        if !timer_started
            timer_started = true
            start_time = now()
            return false
        else
            start_time - now() > duration && :stop
        end
    end
end

timeout(seconds) = timeout(Dates.Second(seconds))


#
#########################



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
      info("Epoch $i")
      $(esc(ex))
    end)
end
