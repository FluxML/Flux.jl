"""
    train!(loss, model, data, opt::FluxState)

Flux 0.14 no longer uses Zygote's implicit parameter dictionary `Flux.params`.

The major change to `train!` is that instead of `loss` being a function which typically accepts
two arguments (the input `x` and expected output `y` from each element of `data`)
now it should typically accept three, the first of which is the `model` itself.

For example, with these definitions...
``` 
data = [(x1, y1), (x2, y2), (x3, y3)];  # each element must be a tuple (or NamedTuple)

loss(m, x, y) = Flux.crossentropy(m(x), y)  # the model is the first argument

opt = Flux.Adam()  # now returns a FluxState
```
...calling `train!(loss, model, data, opt)` runs a loop like this:
```
for d in data
    ∂L∂m = Zygote.gradient(loss, model, d...)[1]
    # update the model using opt & ∂L∂m
end
```
which evaluates the gradient of `loss(model, x1, y1)` with respect to `model`,
to know how to update the parameters stored within `model`.

It is often convenient to provide the function `loss` using `do` block syntax,
instead of defining a named function:
```
Flux.train!(model, Iterators.take(Iterators.cycle(data), 10), Flux.Adam()) do m, x, y
    Flux.crossentropy(m(x), y)  # this does not depend on global variables!
end
```
Here `Iterators.take ∘ Iterators.cycle` uses the same `data` for 10 epochs.

Callback functions are not supported. But see 3-argument `train!` for an
easy way to construct more complicated training loops. For example, this
adds printing & an early stop to the above:
```
for (i, d) in enumerate(data)  
    x, y = d
    ell = Flux.train!(model, opt) do m
        Flux.crossentropy(m(x), y)
    end
    i%10==0 && println("on step \$i, the loss was \$l")  # prints every 10th step
    ell<0.1 && break  # stops training
end
```
"""
function train!(loss::Function, model, data, opt::FluxState)
  _initialise!(opt, model)
  losses = Float32[]
  s = opt.state
  s isa IdDict && error("""Can't mix explicit & implicit modes!
                           Once `FluxState` is initialised by `train!` in one mode, it cannot be used in the other.""")
  for d in data
    l, (g, _...) = explicit_withgradient(loss, model, data_splat(d)...)
    s, model = Optimisers.update!(s, model, g)
    push!(losses, l)
    opt.state = s
  end
  return losses  # Not entirely sure returning losses is a good idea. Flux 0.13 returns `nothing`.
end

data_splat(x::T) where T =  error("""train! expects every d in data be a Tuple or a NamedTuple, got $T
                                   To allow this type, define `Flux.Train.data_splat(x::$T) = (x,)`""")
data_splat(x::Tuple) = x
data_splat(x::NamedTuple) = x

function _initialise!(opt::FluxState, model)
  if opt.state isa Missing
    opt.state = Optimisers.setup(opt.rule, model)
    fmap(model, exclude = Optimisers.isnumeric) do x
      Optimisers.maywrite(x) || error("""model must be fully mutable for train! to work, got x::$(typeof(x))
                                         If `x .+= dx` is in fact ok, define `Optimisers.maywrite(::$(typeof(x))) = true`""")
    end
  end
  opt
end

"""
    train!(loss, model, opt)

While the 4-argument method of `train!` iterates over a dataset,
calling `gradient` many times, this 3-argument version is for a single datapoint,
and calls `gradient` just once.

Its expects a function `loss` which takes just one argument, the model.
For instance:
```
opt = Flux.Adam()
train!(model, opt) do m           # the model is explicitly passed to the function as `m`
    Flux.crossentropy(m(x1), y1)  # but the data point `(x1, y1)` is closed over.
end
```
This calls `Zygote.withgradient(m -> Flux.crossentropy(m(x1), y1), model)`.
(The `do` block is another syntax for this anonymous function.)
Then it updates the parameters contained within `model` according
to the chosen `opt`imiser.
Finally it returns the value of the loss function.
"""
function train!(loss::Function, model, opt::FluxState)
  _initialise!(opt, model)
  s = opt.state
  s isa IdDict && error()
  l, (g, _...) = explicit_withgradient(loss, model)
  opt.state, model = Optimisers.update!(s, model, g)
  l
end

# This method lets you use Optimisers.Descent() instead of Flux.Descent(), when there is no state
function train!(loss::Function, model, data, opt::Optimisers.AbstractRule)
  _initialise!(opt, model)
  fmap(opt.state, exclude = x -> x isa Optimsers.Leaf) do leaf
    leaf.state isa Nothing ||  @warn "Optimiser state will be lost! Please wrap optimisation rule in `FluxState`, e.g. by using `Flux.Adam()`" leaf
    leaf
  end
  train!(loss, model, data, FluxState(opt))
end
