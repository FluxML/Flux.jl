# Flux

*... Initialising Photon Beams ...*

Flux is a library for machine learning, implemented in Julia. In a nutshell, it simply lets you run normal Julia code on a backend like TensorFlow. It also provides many conveniences for doing deep learning.

Flux is very flexible. You can use a convenient Keras-like API if you want something simple, but you can also drop down to straight mathematics, or build your own abstractions. You can even use Flux's utilities (like optimisers) with a completely different backend (like [Knet](https://github.com/denizyuret/Knet.jl)) or mix and match approaches.

Note that Flux is in alpha. Many things work but the API is still in a state of... well, it might change.

**Note:** If you're using Julia v0.5 please see [this version](http://mikeinnes.github.io/Flux.jl/v0.1.1/) of the docs instead.

## Where do I start?

*... Charging Ion Capacitors ...*

The [examples](examples/logreg.html) give a feel for high-level usage. This a great way to start if you're a relative newbie to machine learning or neural networks; you can get up and running running easily.

If you have more experience with ML, or you just don't want to see *those digits* again, check out the [model building guide](models/basics.html) instead. The guide attempts to show how Flux's abstractions are built up and why it's powerful, but it's not all necessary to get started.

## Installation

*... Inflating Graviton Zeppelins ...*

```julia
Pkg.update()
Pkg.add("Flux.jl")
```

You'll also need a backend to run real training, if you don't have one already. Choose from [MXNet](https://github.com/dmlc/MXNet.jl) or [TensorFlow](https://github.com/malmaud/TensorFlow.jl) (MXNet is the recommended option if you're not sure):

```julia
Pkg.add("MXNet") # or "TensorFlow"
Pkg.test("Flux") # Make sure everything installed properly
```

**Note:** TensorFlow integration may not work properly on Julia v0.6 yet.
