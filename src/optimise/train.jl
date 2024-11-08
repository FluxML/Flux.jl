<<<<<<< HEAD
=======
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
>>>>>>> 1466ba36 (let Flux own the function update! to avoid piracy)
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
