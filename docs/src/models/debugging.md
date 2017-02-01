# Debugging Models

Let's take our two-layer perceptron as an example again, running on MXNet:

```julia
@net type TLP
  first
  second
  function (x)
    l1 = σ(first(x))
    l2 = softmax(second(l1))
  end
end

model = TLP(Affine(10, 20), Affine(21, 15))

mxmodel = mxnet(model, (1, 20))
```

Unfortunately, this model has a (fairly obvious) typo, which means that the code above won't run. Instead we get an error message:

```julia
InferShape Error in dot5: [20:37:39] src/operator/./matrix_op-inl.h:271: Check failed: (lshape[1]) == (rshape[0]) dot shape error: (15,21) X (20,1)
 in Flux.Affine at affine.jl:8
 in TLP at test.jl:6
 in mxnet(::TLP, ::Tuple{Int64,Int64}) at model.jl:40
 in mxnet(::TLP, ::Vararg{Any,N} where N) at backend.jl:20
```

Most frameworks would only give the error message here – not so helpful if you have thousands of nodes in your computational graph. However, Flux is able to give good error reports *even when no Julia code has been run*, e.g. when running on a backend like MXNet. This enables us to pinpoint the source of the error very quickly even in a large model.

In this case, we can immediately see that the error occurred within an `Affine` layer. There are two such layers, but this one was called from the second line of `TLP`, so it must be the second `Affine` layer we defined. The layer expected an input of length 21 but got 20 instead.

Of course, often a stack trace isn't enough to figure out the source of an error. Another option is to simply step through the execution of the model using Gallium. While handy, however, stepping isn't always the best way to get a "bird's eye view" of the code. For that, Flux provides a macro called `@shapes`:

```julia
julia> @shapes model(rand(5,10))

# /Users/mike/test.jl, line 18:
gull = σ(Affine(10, 20)(Input()[1]::(5,10))::(5,20))::(5,20)
# /Users/mike/.julia/v0.6/Flux/src/layers/affine.jl, line 8:
lobster = gull * _::(21,15) + _::(1,15)
# /Users/mike/test.jl, line 19:
raven = softmax(lobster)
```

This is a lot like Julia's own `code_warntype`; but instead of annotating expressions with types, we display their shapes. As a lowered form it has some quirks; input arguments are represented by `Input()[N]` and parameters by an underscore.

This makes the problem fairly obvious. We tried to multiply the output of the first layer `(5, 20)` by a parameter `(21, 15)`; the inner dimensions should have been equal.

Notice that while the first `Affine` layer is displayed as-is, the second was inlined and we see a reference to where the `W * x + b` line was defined in Flux's source code. In this way Flux makes it easy to drill down into problem areas, without showing you the full graph of thousands of nodes at once.
