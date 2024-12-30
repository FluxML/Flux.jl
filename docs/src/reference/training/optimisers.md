```@meta
CurrentModule = Flux
CollapsedDocStrings = true
```

# [Optimisation Rules](@id man-optimisers)

Any optimization rule from Optimisers.jl can be used with [`train!`](@ref Flux.Train.train!) and
other training functions.

For full details of how the interface works, see the [Optimisers.jl documentation](https://fluxml.ai/Optimisers.jl/).


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
Optimisers.Lion
```

## Composing Optimisers

Flux (through Optimisers.jl) defines a special kind of optimiser called `OptimiserChain` which takes in arbitrary optimisers as input. Its behaviour is similar to the usual optimisers, but differs in that it acts by calling the optimisers listed in it sequentially. Each optimiser produces a modified gradient
that will be fed into the next, and the resultant update will be applied to the parameter as usual. A classic use case is where adding decays is desirable. Optimisers.jl defines the basic decay corresponding to an $L_2$ regularization in the loss as `WeightDecay`.

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

```@docs
Optimisers.OptimiserChain
```

## Decays

Similar to optimisers, Flux also defines some simple decays that can be used in conjunction with other optimisers, or standalone.

```@docs
Optimisers.SignDecay
Optimisers.WeightDecay
```

## Gradient Clipping

Gradient clipping is useful for training recurrent neural networks, which have a tendency to suffer from the exploding gradient problem. An example usage is

```julia
opt = OptimiserChain(ClipGrad(1e-3), Adam(1e-3))
```

```@docs
Optimisers.ClipGrad
Optimisers.ClipNorm
```


