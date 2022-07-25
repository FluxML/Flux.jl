"""
    train!(loss, pars::Params, data, opt::FluxState)
    
Legacy method, mimicking the behaviour of Flux <= 0.13.
(Note that the implementation is different, using Optimisers.jl internally.)

For each `d in data`, first the gradient of the `loss` is computed like this:
```
    gradient(() -> loss(d...), pars)  # if d isa Tuple
    gradient(() -> loss(d), pars)     # otherwise
```
Here `pars` is produced by calling [`Flux.params`](@ref) on your model.
This is Zygote's "implicit" style of parameter handling.

Then, this gradient is used by optimizer `opt` to update the paramters:
```
    update!(opt, pars, grads)
```
The `data` is iterated through once in this manner.

Typically `data` contains tuples, like `data = [(x1, y1), (x2, y2), (x3, y3)]`.
In this case the function might be `loss(x, y) = mse(model(x), y)`, accepting two arguments.
Notice that it closes over the `model`, which is a global variable.
"""
function train!(loss::Function, pars::Params, data, opt::FluxState)
  Base.depwarn("""`Flux.train!` accepting implicit `Params` is a legacy method in Flux 0.14.
                  Explicit parameters are now preferred, see `train!(loss, model, data, opt)`""", :train!, force=true)
  _initialise!(opt, pars)
  losses = Float32[]
  for d in data
    l, grads = Zygote.withgradient(() -> loss(batchmemaybe(d)...), pars)
    update!(opt, pars, grads)
    push!(losses, l)
  end
  return losses
end

batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x

"""
    train!(loss, pars::Params, opt::FluxState)

This 3-arg method is a bit of a hybrid. With no `data` to iterate over,
it calls `gradient(() -> loss(), pars)` just once, then updates parameters.
"""
function train!(loss::Function, pars::Params, opt::FluxState)
  Base.depwarn("""`Flux.train!` accepting implicit `Params` is a legacy method in Flux 0.14.
                  Explicit parameters are now preferred, see `train!(loss, model, data, opt)`""", :train!, force=true)
  _initialise!(opt, pars)
  l, grads = Zygote.withgradient(() -> loss(), pars)
  update!(opt, pars, grads)
  return l
end

function _initialise!(opt::FluxState, pars::Params)
  dict = IdDict()
  for p in pars
    dict[p] = Optimisers.setup(opt.rule, p)
  end
  opt.state = dict
end

"""
    Flux.update!(opt::FluxState, ps::Params, gs)
    
Legacy method, mimicking the behaviour of Flux <= 0.13.
"""
function update!(opt::FluxState, xs::Params, gs)
  Base.depwarn("Flux.update! is a legacy function", :update!)
  for x in xs
    isnothing(gs[x]) && continue
    update!(opt, x, gs[x])
  end
end

function update!(opt::FluxState, x::AbstractArray, dx)
  opt.state[x], xnew = Optimisers.update!(opt.state[x], x, dx)
  xnew === x || error("failed to mutate x!")
  nothing
end