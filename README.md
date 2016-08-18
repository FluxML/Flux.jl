# Флукс

## What?

Flux is a programming model for building neural networks, implemented in Julia.

## Why?

Flux is designed to be much more intuitive. For starters, that means having a simple notation for models that's as close to the mathematical description as possible (like `σ(W*x + b)`). More importantly, Flux is fully declarative, so there's no more mental juggling of multiple execution paths as you read imperative graph-building code.

Most frameworks intrinsically couple the model (what you'd find in a paper) with its implementation (details like batching and loop unrolling). This greatly increases the overhead involved in both getting a model to work and changing it afterwards. Flux's solution is to distinguish between a *description* of a model and the model itself, just like the class/object distinction. Once you instantiate a model you can alter its implementation as simply as with a call to `batch(model, 100)` or `unroll(model, 10)`.

Flux natively supports for recurrent loops, which it can automatically unroll for you – never do it by hand again.

It's also designed to be extremely flexible. Flux supports multiple backends – MXNet to begin with and TensorFlow in future – transparently taking advantage of all their features rather than providing a lowest common denominator. Flux's design allows for custom layer types – say custom GPU kernels – to be implemented in pure Julia, for backends that support it.

## How?

See [the design docs](design.md).

## Is it any good?

Yes.
