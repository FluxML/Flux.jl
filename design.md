# Flux

Flux tries to provide the best of both worlds from do-it-yourself frameworks like Torch/NN and do-it-for-you frameworks like Keras. It has much in common with, and much different from, both.

At the core is the abstract type `Model`, which is analogous to Torch's `module` – essentially, it's a function which (a) has some internal state and (b) can be differentiated and update its state accordingly.

```julia
model(x) -> y   # Map input -> output (e.g. image -> classification)
back!(model, ∇) # Back-propagate and accumulate errors for the parameters
update!(model)  # Update the model's parameters using the accumulated errors
```

That's it! The `Model` abstraction extends upwards in a nice way – that is, you can stack a bunch of models together like pancakes and you just get a more powerful `Model` back, which can then be reused in the same way.

(It extends downwards, too. Elementary functions like `exp` or `*` a really just `Model`s with zero parameters. Turtles all the way up, turtles all the way down.)

So far this is all very Torch-esque. The downside of Torch's DIY philosophy is that you have to take care of managing memory, pre-allocating temporaries, differentiation and so on yourself. In Flux, however, we can define a type like this:

```julia
@flux type Perceptron <: Model
  W; b
  x -> σ( W * x + b )
end

Perceptron(in::Integer, out::Integer) =
  Perceptron(randn(out, in), randn(out))
```

We've defined a simple Julia type with a couple of parameters, and added a convenient constructor in the usual way. We also defined what should happen when the model is called with an input vector `x`.

The difference is that the `back!` and `update!` functions are now defined for `Perceptron` objects. Flux differentiated the `σ( W * x + b )` expression automatically and figured out handling of temporaries and so on. That's forty or so lines of code that you *could* have written yourself, but it's much nicer not to have to – and the benefits multiply with more complex layers.

Like symbolic frameworks, then, we aim for a very declarative way of defining new layers and architectures. The key difference is that we don't require *all* computation to happen in an opaque, custom runtime. In contrast, Flux simply writes a little code for you and then gets out of the way, making it very easy to understand and extend.

## Recurrence

What really sets Flux apart is how easy it makes it to compose models together in some arbitrary graph. For one thing this makes it very easy to express network architecture; splits, merges, networks running in parallel and so on. But it's also great for recurrence:

```julia
@flux type Recurrent
  Wxh; Whh; Bh
  Wxy; Why; By

  function (x)
    hidden = σ( Wxh*x + Whh*hidden + Bh )
    σ( Wxy*x + Why*hidden′ + By )
  end
end
```

Above, `hidden` is a variable that depends on itself; it creates a cycle in the network. Flux can resolve this cycle by unrolling the network in time.

[recurrence is still very preliminary, I haven't worked out the details of the design yet.]
