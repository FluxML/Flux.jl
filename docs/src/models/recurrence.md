# Recurrent Models

## Recurrent Cells

In the simple feedforward case, our model `m` is a simple function from various inputs `xᵢ` to predictions `yᵢ`. (For example, each `x` might be an MNIST digit and each `y` a digit label.) Each prediction is completely independent of any others, and using the same `x` will always produce the same `y`.

```julia
y₁ = f(x₁)
y₂ = f(x₂)
y₃ = f(x₃)
# ...
```

Recurrent networks introduce a *hidden state* that gets carried over each time we run the model. The model now takes the old `h` as an input, and produces a new `h` as output, each time we run it.

```julia
h = # ... initial state ...
h, y₁ = f(h, x₁)
h, y₂ = f(h, x₂)
h, y₃ = f(h, x₃)
# ...
```

Information stored in `h` is preserved for the next prediction, allowing it to function as a kind of memory. This also means that the prediction made for a given `x` depends on all the inputs previously fed into the model.

(This might be important if, for example, each `x` represents one word of a sentence; the model's interpretation of the word "bank" should change if the previous input was "river" rather than "investment".)

Flux's RNN support closely follows this mathematical perspective. The most basic RNN is as close as possible to a standard `Dense` layer, and the output is also the hidden state.

```julia
Wxh = randn(5, 10)
Whh = randn(5, 5)
b   = randn(5)

function rnn(h, x)
  h = tanh.(Wxh * x .+ Whh * h .+ b)
  return h, h
end

x = rand(10) # dummy data
h = rand(5)  # initial hidden state

h, y = rnn(h, x)
```

If you run the last line a few times, you'll notice the output `y` changing slightly even though the input `x` is the same.

We sometimes refer to functions like `rnn` above, which explicitly manage state, as recurrent *cells*. There are various recurrent cells available, which are documented in the [layer reference](layers.md). The hand-written example above can be replaced with:

```julia
using Flux

rnn2 = Flux.RNNCell(10, 5)

x = rand(10) # dummy data
h = rand(5)  # initial hidden state

h, y = rnn2(h, x)
```

## Stateful Models

For the most part, we don't want to manage hidden states ourselves, but to treat our models as being stateful. Flux provides the `Recur` wrapper to do this.

```julia
x = rand(10)
h = rand(5)

m = Flux.Recur(rnn, h)

y = m(x)
```

The `Recur` wrapper stores the state between runs in the `m.state` field.

If you use the `RNN(10, 5)` constructor – as opposed to `RNNCell` – you'll see that it's simply a wrapped cell.

```julia
julia> RNN(10, 5)
Recur(RNNCell(10, 5, tanh))
```

## Sequences

Often we want to work with sequences of inputs, rather than individual `x`s.

```julia
seq = [rand(10) for i = 1:10]
```

With `Recur`, applying our model to each element of a sequence is trivial:

```julia
m.(seq) # returns a list of 5-element vectors
```

This works even when we've chain recurrent layers into a larger model.

```julia
m = Chain(LSTM(10, 15), Dense(15, 5))
m.(seq)
```

## Truncating Gradients

By default, calculating the gradients in a recurrent layer involves its entire history. For example, if we call the model on 100 inputs, we'll have to calculate the gradient for those 100 calls. If we then calculate another 10 inputs we have to calculate 110 gradients – this accumulates and quickly becomes expensive.

To avoid this we can *truncate* the gradient calculation, forgetting the history.

```julia
truncate!(m)
```

Calling `truncate!` wipes the slate clean, so we can call the model with more inputs without building up an expensive gradient computation.

`truncate!` makes sense when you are working with multiple chunks of a large sequence, but we may also want to work with a set of independent sequences. In this case the hidden state should be completely reset to its original value, throwing away any accumulated information. `reset!` does this for you.

In general, when training with recurrent layers in your model, you'll want to call `reset!` or `truncate!` for each loss calculation:

```julia
function loss(x,y)
  l = Flux.mse(m(x), y)
  Flux.reset!(m)
  return l
end
```

## Learning the Initial State

By default, the hidden state of RNN cells is initialised to zero, and is reset to zero every time `Flux.reset!` is called. Some models treat this initial state as a learnable parameter, which must be optimised during training. For such models, *learn_initial_state!* must be invoked:

```julia
Flux.learn_initial_state!(m)
```
