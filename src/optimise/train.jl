function update!(opt::AbstractOptimiser, x::AbstractArray, x̄)
  x̄r = copyto!(similar(x̄), x̄)  # Flux.Optimise assumes it can mutate the gradient. This is not
                               # safe due to aliasing, nor guaranteed to be possible, e.g. Fill.
  x .-= apply!(opt, x, x̄r)
end

function update!(opt::AbstractOptimiser, xs::Params, gs)
  @warn """The method `Flux.update!(optimiser, ps::Params, grads)` is deprecated,
      as part of Flux's move away from Zyote's implicit mode.
      Please use explicit-style `update!(opt_state, model, grad)` instead,
      where `grad = Flux.gradient(m -> loss(m,x,y), model)` and `opt_state = Flux.setup(rule, model)`.""" maxlog=1
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
  @warn """The method `Flux.train!(loss2, ps::Params, data, optimiser)` is deprecated,
    as part of Flux's move away from Zyote's implicit parameters.
    Please use explicit-style `train!(loss, model, data, opt_state)` instead,
    where `loss(m, xy...)` accepts the model, and `opt_state = Flux.setup(rule, model)`.""" maxlog=1
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
