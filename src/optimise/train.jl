using ProgressLogging: @progress, @withprogress, @logprogress
import Zygote: Params, gradient

"""
    update!(x, x̄)

Update the array `x` according to `x .-= x̄`.
"""
function update!(x::AbstractArray, x̄)
  x .-= x̄
end

"""
    update!(opt, p, g)
    update!(opt, ps::Params, gs)

Perform an update step of the parameters `ps` (or the single parameter `p`)
according to optimizer `opt`  and the gradients `gs` (the gradient `g`).

As a result, the parameters are mutated and the optimizer's internal state may change.
The gradient could be mutated as well.
"""
function update!(opt, x, x̄)
  x̄r = ArrayInterface.restructure(x, x̄) # address some cases where Zygote's
                                          # output are not mutable, see #1510 
  x .-= apply!(opt, x, x̄r)
end

function update!(opt, xs::Params, gs)
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

Call `Flux.skip()` within the loss passed to [`train!`](@ref) to exit the
current calculation and `continue` the training loop with the next data point.

This can be used (say) to avoid updating model parameters for those data points
whose loss is infinite, or NaN. It works by throwing an error which is caught by `train!`.

Note that calling this from a callback is almost useless, as `cb` is evaluated
after the `gradient` and `update!` steps have been performed.

# Examples
```jldoctest
julia> model = [10.5f0];

julia> Flux.train!(Flux.params(model), 1:10, Descent(1f0)) do i
          @show i
          3 < i < 9 && Flux.skip()
          @show sum(model)
       end
i = 1
sum(model) = 10.5f0
i = 2
sum(model) = 9.5f0
i = 3
sum(model) = 8.5f0
i = 4
i = 5
i = 6
i = 7
i = 8
i = 9
sum(model) = 7.5f0
i = 10
sum(model) = 6.5f0

julia> model
1-element Vector{Float32}:
 5.5
```
"""
function skip()
  throw(SkipException())
end


struct StopException <: Exception end

"""
    stop()

Call `Flux.stop()` in a callback passed to [`train!`](@ref), 
or inside the loss, to immediately `break` out of the training loop.

This can be used to stop training when some accuracy is achieved.
(It throws an error which is caught by `train!`.)

# Example
```jldoctest
julia> model = [10.5f0];

julia> cb = () -> sum(model) < 8 && Flux.stop();

julia> Flux.train!(Flux.params(model), 1:10, Descent(1f0); cb) do i
         @show i sum(model)
       end
i = 1
sum(model) = 10.5f0
i = 2
sum(model) = 9.5f0
i = 3
sum(model) = 8.5f0

julia> model
1-element Vector{Float32}:
 7.5
```
"""
function stop()
  throw(StopException())
end

batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x

"""
    train!(loss, params, data, opt; cb)
        
`train!` uses a `loss` function and training `data` to improve the 
[Model parameters](@ref) (`params`) based on a pluggable [Optimisers](@ref) (`opt`).
        
For each datapoint `d` in `data`, compute the gradient of  `loss` with
respect to `params` through backpropagation and call the optimizer `opt`.
If `d` is a tuple of arguments to `loss` call `loss(d...)`, else call `loss(d)`.
        
To pass trainable parameters, call [`Flux.params`](@ref) with your model or just the 
layers you want to train, like `train!(loss, params(model), ...)` or `train!(loss, params(model[1:end-2), ...)` respectively.

[Callbacks](@ref) are given with the keyword argument `cb`. For example, this will print "training" 
every 10 seconds (using [`Flux.throttle`](@ref)):
`train!(loss, params, data, opt, cb = throttle(() -> println("training"), 10))`
        
The callback can call [`Flux.stop`](@ref) to interrupt the training loop.

Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function train!(loss, ps, data, opt; cb = () -> ())
  ps = Params(ps)
  cb = runall(cb)
  n = (Base.IteratorSize(typeof(data)) == Base.HasLength()) ? length(data) : 0
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
    @logprogress i / n
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
  :(@progress for i = 1:$(esc(n))
      @info "Epoch $i"
      $(esc(ex))
    end)
end
