# Флукс

## What?

Flux is an experimental machine perception / ANN library for Julia. It's designed to make experimenting with novel layer types and architectures really fast, without sacrificing runtime speed.

## Why?

Flux has a few key differences from other libraries:

* Flux's [graph-based DSL](https://github.com/MikeInnes/Flow.jl), which provides optimisations and automatic differentiation, is very tightly integrated with the language. This means nice syntax for your equations (`σ(W*x+b)` anyone?) and no unwieldy `compile` steps.
* The graph DSL directly is used to represent models (not just computations), so custom architectures – and in particular, recurrent models – are easy to express.
* Those fancy features are completely optional. You can implement functionality in a Torch-like fashion if you wish, since layers are simply objects that satisfy a small interface.
* Flux is written in [Julia](http://julialang.org), which means there's no "dropping down" to C. It's Julia all the way down, and you can prototype both high-level architectures and high-performance GPU kernels from the same language. This also makes the library itself very easy to understand and extend.

Future work will also include:

* Integration with other backends, so that models can be described using Flux and run using (say) TensorFlow.
* Carrying out runtime optimisations of the graph, in particular to handle small matrices efficiently.

## How?

See [the design docs](design.md).

## Is it any good?

Yes.
