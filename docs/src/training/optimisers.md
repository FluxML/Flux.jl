```@meta
CurrentModule = Flux
```

# [Optimisation Rules](@id man-optimisers)

Any optimization rule from Optimisers.jl can be used with [`train!`](@ref) and
other training functions.

For full details of how the new interface works, see the [Optimisers.jl documentation](https://fluxml.ai/Optimisers.jl/dev/).


## Optimisers Reference

All optimisers return an object that, when passed to `train!`, will update the parameters passed to it.

```@docs
Optimisers.Descent
Optimisers.Momentum
Optimisers.Nesterov
Optimisers.RMSProp
Optimisers.Adam
Optimisers.RAdam
Optimisers.AdaMax
Optimisers.AdaGrad
Optimisers.AdaDelta
Optimisers.AMSGrad
Optimisers.NAdam
Optimisers.AdamW
Optimisers.OAdam
Optimisers.AdaBelief
```

## Composing Optimisers

Flux (through Optimisers.jl) defines a special kind of optimiser called `OptimiserChain` which takes in arbitrary optimisers as input. Its behaviour is similar to the usual optimisers, but differs in that it acts by calling the optimisers listed in it sequentially. Each optimiser produces a modified gradient
that will be fed into the next, and the resultant update will be applied to the parameter as usual. A classic use case is where adding decays is desirable. Optimisers.jl defines the basic decay corresponding to an $L_2$ regularization in the loss as `WeighDecay`.

```julia
opt = OptimiserChain(WeightDecay(1e-4), Descent())
```

Here we apply the weight decay to the `Descent` optimiser. 
The resulting optimiser `opt` can be used as any optimiser.

```julia
w = [randn(10, 10), randn(10, 10)]
opt_state = Flux.setup(opt, w)

loss(w, x) = Flux.mse(w[1] * x, w[2] * x)

loss(w, rand(10)) # around 0.9

for t = 1:10^5
  g = gradient(w -> loss(w[1], w[2], rand(10)), w)
  Flux.update!(opt_state, w, g)
end

loss(w, rand(10)) # around 0.9
```

It is possible to compose optimisers for some added flexibility.

## Scheduling Optimisers

In practice, it is fairly common to schedule the learning rate of an optimiser to obtain faster convergence. There are a variety of popular scheduling policies, and you can find implementations of them in [ParameterSchedulers.jl](http://fluxml.ai/ParameterSchedulers.jl/stable). The documentation for ParameterSchedulers.jl provides a more detailed overview of the different scheduling policies, and how to use them with Flux optimisers. Below, we provide a brief snippet illustrating a [cosine annealing](https://arxiv.org/pdf/1608.03983.pdf) schedule with a momentum optimiser.

First, we import ParameterSchedulers.jl and initialize a cosine annealing schedule to vary the learning rate between `1e-4` and `1e-2` every 10 steps. We also create a new [`Momentum`](@ref) optimiser.
```julia
using ParameterSchedulers

opt = Momentum()
schedule = Cos(λ0 = 1e-4, λ1 = 1e-2, period = 10)
for (eta, epoch) in zip(schedule, 1:100)
  opt.eta = eta
  # your training code here
end
```
`schedule` can also be indexed (e.g. `schedule(100)`) or iterated like any iterator in Julia.

ParameterSchedulers.jl schedules are stateless (they don't store their iteration state). If you want a _stateful_ schedule, you can use `ParameterSchedulers.Stateful`:
```julia
using ParameterSchedulers: Stateful, next!

schedule = Stateful(Cos(λ0 = 1e-4, λ1 = 1e-2, period = 10))
for epoch in 1:100
  opt.eta = next!(schedule)
  # your training code here
end
```

ParameterSchedulers.jl allows for many more scheduling policies including arbitrary functions, looping any function with a given period, or sequences of many schedules. See the ParameterSchedulers.jl documentation for more info.

## Decays

Similar to optimisers, Flux also defines some simple decays that can be used in conjunction with other optimisers, or standalone.

```@docs
SignDecay
WeightDecay
```

## Gradient Clipping

Gradient clipping is useful for training recurrent neural networks, which have a tendency to suffer from the exploding gradient problem. An example usage is

```julia
opt = OptimiserChain(ClipValue(1e-3), Adam(1e-3))
```

```@docs
ClipGrad
ClipNorm
```


