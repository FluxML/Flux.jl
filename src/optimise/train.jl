using Juno
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
"""
function update!(opt, x, x̄)
  x .-= apply!(opt, x, x̄)
end

function update!(opt, xs::Params, gs)
  for x in xs
    gs[x] == nothing && continue
    update!(opt, x, gs[x])
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
This will trigger the train loop to stop and exit.

# Examples
```julia
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

For each datapoint `d` in `data` compute the gradient of `loss(d...)` through
backpropagation and call the optimizer `opt`.

In case datapoints `d` are of numeric array type, assume no splatting is needed
and compute the gradient of `loss(d)`.

A callback is given with the keyword argument `cb`. For example, this will print
"training" every 10 seconds (using [`Flux.throttle`](@ref)):

  train!(loss, params, data, opt,
         cb = throttle(() -> println("training"), 10))

The callback can call [`Flux.stop`](@ref) to interrupt the training loop.

Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function train!(loss, ps, data, opt; cb = () -> (), epochs = 1, steps_per_epoch = typemax(Int), verbose = true)
  ps = Params(ps)
  cb = runall(cb)
  l̄ = 0f0
  cb_narg = methods(cb).ms[1].nargs - 1
  ndata = min(length(data), steps_per_epoch)
  for epoch in 1:epochs
    printstyled("Epoch $epoch/$epochs\n", color = :yellow)
    prog = Progress(ndata)
    for (n, d) in enumerate(data)
      n > steps_per_epoch && break
      try
        local l
        if d isa AbstractArray{<:Number}
          gs = gradient(ps) do
            l = loss(d)
          end
        else
          gs = gradient(ps) do
            l = loss(d...)
          end
        end
        update!(opt, ps, gs)
        l̄ = ((n - 1) * l̄ + l) / n
        if verbose
          prog.desc = "$n/$ndata "
          next!(prog, showvalues = [(:loss, l), (:avgloss, l̄)])
        end
        cb_narg > 0 ? cb(ps, l) : cb()
      catch ex
        if ex isa StopException
          break
        else
          rethrow(ex)
        end
      end
    end
  end
  return l̄
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
