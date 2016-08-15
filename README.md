# Флукс

## What?

Flux is a programming model for building neural networks, implemented in Julia.

## Why?

Flux is designed to be much more intuitive than traditional frameworks. For starters, that means having a simple notation for models that's as close to the mathematical description as possible (like `σ(W*x + b)`). But it's deeper than syntax; we also reuse concepts from regular programming languages (like the class/object distinction) to create principled semantics. Flux is fully declarative, so there's no more mental juggling of multiple execution paths as you read imperative graph-building code.

Flux's semantics include native support for recurrent loops, which it can automatically unroll for you – never do it by hand again.

But it's also designed to be extremely flexible. Flux supports multiple backends – MXNet to begin with and TensorFlow in future – transparently taking advantage of all their features rather than providing a lowest common denominator. Flux's design allows for custom layer types – say custom GPU kernels – to be implemented in pure Julia, for backends that support it.

## How?

See [the design docs](design.md).

## Is it any good?

Yes.
