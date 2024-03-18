using ProgressLogging: @progress, @withprogress, @logprogress
import Zygote: Params, gradient, withgradient

# Add methods to Optimisers.jl's function, so that there is just one Flux.update!
# for both explicit and implicit parameters.
import Optimisers.update!

"""
    update!(opt, p, g)
    update!(opt, ps::Params, gs)

Perform an update step of the parameters `ps` (or the single parameter `p`)
according to optimiser `opt::AbstractOptimiser`  and the gradients `gs` (the gradient `g`).

As a result, the parameters are mutated and the optimiser's internal state may change.
The gradient could be mutated as well.

!!! compat "Deprecated"
    This method for implicit `Params` (and `AbstractOptimiser`) will be removed from Flux 0.15.
    The explicit method `update!(opt, model, grad)` from Optimisers.jl will remain.
"""
function update!(opt::AbstractOptimiser, x::AbstractArray, x̄)
  x̄r = copyto!(similar(x̄), x̄)  # Flux.Optimise assumes it can mutate the gradient. This is not
                               # safe due to aliasing, nor guaranteed to be possible, e.g. Fill.
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


batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x

"""
    train!(loss, pars::Params, data, opt::AbstractOptimiser; [cb])
        
Uses a `loss` function and training `data` to improve the 
model's parameters according to a particular optimisation rule `opt`.

!!! compat "Deprecated"
    This method with implicit `Params` will be removed from Flux 0.15.
    It should be replaced with the explicit method `train!(loss, model, data, opt)`.

For each `d in data`, first the gradient of the `loss` is computed like this:
```
    gradient(() -> loss(d...), pars)  # if d isa Tuple
    gradient(() -> loss(d), pars)     # otherwise
```
Here `pars` is produced by calling [`Flux.params`](@ref) on your model.
(Or just on the layers you want to train, like `train!(loss, params(model[1:end-2]), data, opt)`.)
This is the "implicit" style of parameter handling.

This gradient is then used by optimiser `opt` to update the parameters:
```
    update!(opt, pars, grads)
```
The optimiser should be from the `Flux.Optimise` module (see [Optimisers](@ref)).
Different optimisers can be combined using [`Flux.Optimise.Optimiser`](@ref Flux.Optimiser).

This training loop iterates through `data` once.
It will stop with a `DomainError` if the loss is `NaN` or infinite.

You can use use `train!` inside a for loop to do this several times, or 
use for instance `Itertools.ncycle` to make a longer `data` iterator.

## Callbacks

[Callbacks](@ref) are given with the keyword argument `cb`.
For example, this will print "training" every 10 seconds (using [`Flux.throttle`](@ref)):
```
    train!(loss, params, data, opt, cb = throttle(() -> println("training"), 10))
```

Multiple callbacks can be passed to `cb` as array.
"""
function train!(loss, ps::Params, data, opt::AbstractOptimiser; cb = () -> ())
  cb = runall(cb)
  itrsz = Base.IteratorSize(typeof(data))
  n = (itrsz == Base.HasLength()) || (itrsz == Base.HasShape{1}()) ? length(data) : 0
  @withprogress for (i, d) in enumerate(data)
    l, gs = withgradient(ps) do
      loss(batchmemaybe(d)...)
    end
    if !isfinite(l)
      throw(DomainError(lazy"Loss is $l on data item $i, stopping training"))
    end
    update!(opt, ps, gs)
    cb()

    @logprogress iszero(n) ? nothing : i / n
  end
end
