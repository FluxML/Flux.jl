# Recurrent Models

[Recurrence](https://en.wikipedia.org/wiki/Recurrent_neural_network) is a first-class feature in Flux and recurrent models are very easy to build and use. Recurrences are often illustrated as cycles or self-dependencies in the graph; they can also be thought of as a hidden output from / input to the network. For example, for a sequence of inputs `x1, x2, x3 ...` we produce predictions as follows:

```julia
y1 = f(W, x1) # `f` is the model, `W` represents the parameters
y2 = f(W, x2)
y3 = f(W, x3)
...
```

Each evaluation is independent and the prediction made for a given input will always be the same. That makes a lot of sense for, say, MNIST images, but less sense when predicting a sequence. For that case we introduce the hidden state:

```julia
y1, s = f(W, x1, s)
y2, s = f(W, x2, s)
y3, s = f(W, x3, s)
...
```

The state `s` allows the prediction to depend not only on the current input `x` but also on the history of past inputs.

The simplest recurrent network looks as follows in Flux, and it should be familiar if you've seen the equations defining an RNN before:

```julia
@net mutable struct Recurrent
  Wxy; Wyy; by
  y
  function (x)
    y = tanh( x * Wxy + y{-1} * Wyy + by )
  end
end
```

The only difference from a regular feed-forward layer is that we create a variable `y` which is defined as depending on itself. The `y{-1}` syntax means "take the value of `y` from the previous run of the network".

Using recurrent layers is straightforward and no different feedforward ones in terms of the `Chain` macro etc. For example:

```julia
model = Chain(
    Affine(784, 20), σ
    Recurrent(20, 30),
    Recurrent(30, 15))
```

Before using the model we need to unroll it. This happens with the `unroll` function:

```julia
unroll(model, 20)
```

This call creates an unrolled, feed-forward version of the model which accepts N (= 20) inputs and generates N predictions at a time. Essentially, the model is replicated N times and Flux ties the hidden outputs `y` to hidden inputs.

Here's a more complex recurrent layer, an LSTM, and again it should be familiar if you've seen the [equations](https://colah.github.io/posts/2015-08-Understanding-LSTMs/):

```julia
@net mutable struct LSTM
  Wxf; Wyf; bf
  Wxi; Wyi; bi
  Wxo; Wyo; bo
  Wxc; Wyc; bc
  y; state
  function (x)
    # Gates
    forget = σ( x * Wxf + y{-1} * Wyf + bf )
    input  = σ( x * Wxi + y{-1} * Wyi + bi )
    output = σ( x * Wxo + y{-1} * Wyo + bo )
    # State update and output
    state′ = tanh( x * Wxc + y{-1} * Wyc + bc )
    state  = forget .* state{-1} + input .* state′
    y = output .* tanh(state)
  end
end
```

The only unfamiliar part is that we have to define all of the parameters of the LSTM upfront, which adds a few lines at the beginning.
