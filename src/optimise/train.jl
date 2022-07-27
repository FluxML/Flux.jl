using ProgressLogging: @progress, @withprogress, @logprogress
import Zygote: Params, gradient


"""
    update!(opt, p, g)
    update!(opt, ps::Params, gs)

Perform an update step of the parameters `ps` (or the single parameter `p`)
according to optimizer `opt`  and the gradients `gs` (the gradient `g`).

As a result, the parameters are mutated and the optimizer's internal state may change.
The gradient could be mutated as well.
"""
function update!(opt::AbstractOptimiser, x, x̄)
  x̄r = ArrayInterface.restructure(x, x̄) # address some cases where Zygote's
                                          # output are not mutable, see #1510 
  x .-= apply!(opt, x, x̄r)
end

function update!(opt::AbstractOptimiser, xs::Params, gs)
  for x in xs
    isnothing(gs[x]) && continue
    update!(opt, x, gs[x])
  end
end

# Callback niceties
call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

struct SkipException <: Exception end

"""
    skip()

Call `Flux.skip()` in a callback to indicate when a callback condition is met.
This will trigger the train loop to skip the current data point and not update with the calculated gradient.

# Examples
```julia
cb = function ()
  loss() > 1e7 && Flux.skip()
end
```
"""
function skip()
  Base.depwarn("""Flux.skip() will be removed from Flux 0.14.
                  It never worked as described, 
                  and should be replaced with `continue` in an ordinary `for` loop.""", :skip)
  throw(SkipException())
end


struct StopException <: Exception end

"""
    stop()

Call `Flux.stop()` in a callback to indicate when a callback condition is met.
This will trigger the train loop to stop and exit.

# Examples
```julia
cb = function ()
  accuracy() > 0.9 && Flux.stop()
end
```
"""
function stop()
  Base.depwarn("""Flux.stop() will be removed from Flux 0.14.
                  It should be replaced with `break` in an ordinary `for` loop.""", :stop)
  throw(StopException())
end

batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x

"""
    train!(loss, pars::Params, data, opt::AbstractOptimiser; [cb])
        
Uses a `loss` function and training `data` to improve the 
model's parameters according to a particular optimisation rule `opt`.

For each `d in data`, first the gradient of the `loss` is computed like this:
```
    gradient(() -> loss(d...), pars)  # if d isa Tuple
    gradient(() -> loss(d), pars)     # otherwise
```
Here `pars` is produced by calling [`Flux.params`](@ref) on your model.
(Or just on the layers you want to train, like `train!(loss, params(model[1:end-2]), data, opt)`.)
This is the "implicit" style of parameter handling.

Then, this gradient is used by optimizer `opt` to update the paramters:
```
    update!(opt, pars, grads)
```
The optimiser should be from the [Flux.Optimise](@ref) module.
Different optimisers can be combined using [Flux.Optimise.Optimiser](@ref).

This training loop iterates through `data` once.
You can use [`@epochs`](@ref) to do this several times, or 
use for instance `Iterators.repeat` to make a longer `data` iterator.

## Callbacks

[Callbacks](@ref) are given with the keyword argument `cb`.
For example, this will print "training" every 10 seconds (using [`Flux.throttle`](@ref)):
```
    train!(loss, params, data, opt, cb = throttle(() -> println("training"), 10))
```
    
The callback can call [`Flux.stop`](@ref) to interrupt the training loop.

Multiple callbacks can be passed to `cb` as array.
"""
function train!(loss, ps::Params, data, opt::AbstractOptimiser; cb = () -> ())
  cb = runall(cb)
  itrsz = Base.IteratorSize(typeof(data))
  n = (itrsz == Base.HasLength()) || (itrsz == Base.HasShape{1}()) ? length(data) : 0
  @withprogress for (i, d) in enumerate(data)
    try
      gs = gradient(ps) do
        loss(batchmemaybe(d)...)
      end
      update!(opt, ps, gs)
      cb()
    catch ex
      if ex isa StopException
        break
      elseif ex isa SkipException
        continue
      else
        rethrow(ex)
      end
    end
    @logprogress iszero(n) ? nothing : i / n
  end
end

"""
    @epochs N body

Run `body` `N` times. Mainly useful for quickly doing multiple epochs of
training in a REPL.

# Examples
```jldoctest
julia> Flux.@epochs 2 println("hello")
[ Info: Epoch 1
hello
[ Info: Epoch 2
hello
```
"""
macro epochs(n, ex)
  Base.depwarn("""The macro `@epochs` will be removed from Flux 0.14.
                  Just write an ordinary for loop.""", Symbol("@epochs"), force=true)
  :(@progress for i = 1:$(esc(n))
      @info "Epoch $i"
      $(esc(ex))
    end)
end
