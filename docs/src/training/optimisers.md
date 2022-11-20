```@meta
CurrentModule = Flux
```

# [Optimisers](@id man-optimisers)

Flux builds in many optimisation rules for use with [`train!`](@ref Flux.Optimise.train!) and
other training functions.

The mechanism by which these work is gradually being replaced as part of the change
from "implicit" dictionary-based to "explicit" tree-like structures.
At present, the same struct (such as `Adam`) can be used with either form,
and will be automatically translated.

For full details of how the new "explicit" interface works, see the [Optimisers.jl documentation](https://fluxml.ai/Optimisers.jl/dev/).

For full details on how the "implicit" interface worked, see the [Flux 0.13.6 manual](https://fluxml.ai/Flux.jl/v0.13.6/training/optimisers/#Optimiser-Interface).


## Optimiser Reference

All optimisers return an object that, when passed to `train!`, will update the parameters passed to it.

```@docs
Descent
Momentum
Nesterov
RMSProp
Adam
RAdam
AdaMax
AdaGrad
AdaDelta
AMSGrad
NAdam
AdamW
OAdam
AdaBelief
```

## Composing Optimisers

Flux defines a special kind of optimiser simply called `Optimiser` which takes in arbitrary optimisers as input. Its behaviour is similar to the usual optimisers, but differs in that it acts by calling the optimisers listed in it sequentially. Each optimiser produces a modified gradient
that will be fed into the next, and the resultant update will be applied to the parameter as usual. A classic use case is where adding decays is desirable. Flux defines some basic decays including `ExpDecay`, `InvDecay` etc.

```julia
opt = Optimiser(ExpDecay(1, 0.1, 1000, 1e-4), Descent())
```

Here we apply exponential decay to the `Descent` optimiser. The defaults of `ExpDecay` say that its learning rate will be decayed every 1000 steps.
It is then applied like any optimiser.

```julia
w = randn(10, 10)
w1 = randn(10,10)
ps = Params([w, w1])

loss(x) = Flux.Losses.mse(w * x, w1 * x)

loss(rand(10)) # around 9

for t = 1:10^5
  θ = Params([w, w1])
  θ̄ = gradient(() -> loss(rand(10)), θ)
  Flux.Optimise.update!(opt, θ, θ̄)
end

loss(rand(10)) # around 0.9
```

It is possible to compose optimisers for some added flexibility.

```@docs
Flux.Optimise.Optimiser
```

## Scheduling Optimisers

In practice, it is fairly common to schedule the learning rate of an optimiser to obtain faster convergence. There are a variety of popular scheduling policies, and you can find implementations of them in [ParameterSchedulers.jl](https://darsnack.github.io/ParameterSchedulers.jl/dev/README.html). The documentation for ParameterSchedulers.jl provides a more detailed overview of the different scheduling policies, and how to use them with Flux optimizers. Below, we provide a brief snippet illustrating a [cosine annealing](https://arxiv.org/pdf/1608.03983.pdf) schedule with a momentum optimiser.

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
ExpDecay
InvDecay
WeightDecay
```

## Gradient Clipping

Gradient clipping is useful for training recurrent neural networks, which have a tendency to suffer from the exploding gradient problem. An example usage is

```julia
opt = Optimiser(ClipValue(1e-3), Adam(1e-3))
```

```@docs
ClipValue
ClipNorm
```


