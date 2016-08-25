# Флукс

## What?

Flux is an experimental programming model for building neural networks, implemented in Julia. It's not complete yet but you can check out the current functionality in the `examples` folder.

## Why?

Flux has several design goals. Firstly, it's designed to be extremely intuitive. It has a simple notation for models that's as close to the mathematical description as possible (like `σ(W*x + b)`). It's fully declarative, so there's no mental juggling of multiple execution paths as you read imperative graph-building code.

Flux decouples the model (what you'd find in a paper) from the implementation (details like batching and unrolling), increasing flexibility when defining and modifying models. It provides functions which operate over entire models at once, which means you can alter the implementation as simply as with a call to `batch(model, 100)` or `unroll(model, 10)`. (And yes, Flux natively supports recurrent loops, which it can automatically unroll for you – never do it by hand again.)

It's also designed to be extremely flexible. Flux supports multiple backends (like MXNet and TensorFlow) and can transparently take advantage of features unique to the backend. Custom layer types can be implemented in pure Julia, and you can even mix and match different backends together.

Finally, Flux is hackable. Using Julia enables custom kernels, including GPU code, to be written in an interactive and high-level way. Flux's whole implementation – including all built-in layers and utilities – is under 500 lines of pure Julia code.

## How?

See [docs](/docs).

## Is it any good?

Yes.
