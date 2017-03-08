# Backends

## Basic Usage

```julia
model = Chain(Affine(10, 20), σ, Affine(20, 15), softmax)
xs = rand(10)
```

Currently, Flux's pure-Julia backend has no optimisations. This means that calling

```julia
model(rand(10)) #> [0.0650, 0.0655, ...]
```

directly won't have great performance. In order to run a computationally intensive training process, we rely on a backend like MXNet or TensorFlow.

This is easy to do. Just call either `mxnet` or `tf` on a model to convert it to a model of that kind:

```julia
mxmodel = mxnet(model)
mxmodel(xs) #> [0.0650, 0.0655, ...]
# or
tfmodel = tf(model)
tfmodel(xs) #> [0.0650, 0.0655, ...]
```

These new models look and feel exactly like every other model in Flux, including returning the same result when you call them, and can be trained as usual using `Flux.train!()`. The difference is that the computation is being carried out by a backend, which will usually give a large speedup.

## Native Integration

Flux aims to provide high-level APIs that work well across backends, but in some cases you may want to take advantage of features specific to a given backend. In these cases it's easy to "drop down" and use the backend's API directly, where appropriate. For example:

```julia
using MXNet
Flux.loadmx()

mxmodel = mx.FeedForward(model)
```

This returns a standard `mx.FeedForward` instance, just like you might have created using MXNet's usual API. You can then use this with MXNet's data provider implementation, custom optimisers, or distributed training processes.

Same goes for TensorFlow, where it's easy to create a `Tensor` object:

```julia
using TensorFlow
Flux.loadtf()

x  = placeholder(Float32)
y = Tensor(model, x)
```

This makes makes it easy to take advantage of Flux's model description and debugging tools while also getting the benefit of the work put into these backends. You can check out how this looks with the integration examples [here](https://github.com/MikeInnes/Flux.jl/tree/master/examples).
