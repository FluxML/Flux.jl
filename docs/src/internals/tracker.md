# Flux.Tracker

Backpropagation, or reverse-mode automatic differentiation, is handled by the `Flux.Tracker` module.

```julia
julia> using Flux.Tracker
```

Here we discuss some more advanced uses of this module, as well as covering its internals.

## Taking Gradients

In the [basics section](../models/basics.md) we covered basic usage of the `gradient` function.

```julia
using Flux.Tracker

Tracker.gradient((a, b) -> a*b, 2, 3) # (3.0 (tracked), 2.0 (tracked))
```

`gradient` is actually just a thin wrapper around the backpropagator-based interface, `forward`.

```julia
using Flux.Tracker: forward

y, back = forward((a, b) -> a*b, 2, 3) # (6.0 (tracked), Flux.Tracker.#9)

back(1) # (3.0 (tracked), 2.0 (tracked))
```

The `forward` function returns two results. The first, `y`, is the original value of the function (perhaps with tracking applied). The second, `back`, is a new function which, given a sensitivity, returns the sensitivity of the inputs to `forward` (we call this a "backpropagator"). One use of this interface is to provide custom sensitivities when outputs are not scalar.

```julia
julia> y, back = forward((a, b) -> a.*b, [1,2,3],[4,5,6])
(param([4.0, 10.0, 18.0]), Flux.Tracker.#9)

julia> back([1,1,1])
(param([4.0, 5.0, 6.0]), param([1.0, 2.0, 3.0]))
```

We can also take gradients in-place. This can be useful if you only care about first-order gradients.

```julia
a, b = param(2), param(3)

c = a*b # 6.0 (tracked)

Tracker.back!(c)

Tracker.grad(a), Tracker.grad(b) # (3.0, 2.0)
```

## Tracked Arrays

The `param` function converts a normal Julia array into a new object that, while behaving like an array, tracks extra information that allows us to calculate derivatives. For example, say we multiply two parameters:

```julia
julia> W = param([1 2; 3 4])
Tracked 2×2 Array{Float64,2}:
 1.0  2.0
 3.0  4.0

julia> x = param([5, 6])
Tracked 2-element Array{Float64,1}:
 5.0
 6.0

julia> y = W*x
Tracked 2-element Array{Float64,1}:
 17.0
 39.0
```

The output `y` is also a `TrackedArray` object. We can now backpropagate sensitivities to `W` and `x` via the `back!` function, and see the gradients accumulated in the `W` and `x` tracked arrays:

```julia
julia> Tracker.back!(y, [1, -1])

julia> W.grad
2×2 Array{Float64,2}:
 5.0   6.0
-5.0  -6.0

julia> x.grad
2-element Array{Float64,1}:
 -2.0
 -2.0
```

You may sometimes want to drop derivative information and just get the plain value back. You can do this by calling `Tracker.data(W)`.

## Custom Gradients

We can hook in to the processes above to implement custom gradients for a function or kernel. For a toy example, imagine a custom implementation of `minus`:

```julia
minus(a, b) = a - b
```

Firstly, we must tell the tracker system to stop when it sees a call to `minus`, and record it. We can do this using dispatch:

```julia
using Flux.Tracker: TrackedReal, track, @grad

minus(a::TrackedReal, b::TrackedReal) = Tracker.track(minus, a, b)
```

`track` takes care of building a new `Tracked` object and recording the operation on the tape. We just need to provide a gradient definition.

```julia
@grad function minus(a, b)
  return minus(a.data, b.data), Δ -> (Δ, -Δ)
end
```

This is essentially just a way of overloading the `forward` function we saw above. We strip tracking from `a` and `b` so that we are calling the original definition of `minus` (otherwise, we'd just try to track the call again and hit an infinite regress).

Note that in the backpropagator we don't call `data(a)`; we *do* in fact want to track this, since nest AD will take a derivative through the backpropagator itself. For example, the gradient of `*` might look like this.

```julia
@grad a * b = data(a)*data(b), Δ -> (Δ*b, a*Δ)
```

For multi-argument functions with custom gradients, you likely want to catch not just `minus(::TrackedArray, ::TrackedArray)` but also `minus(::Array, TrackedArray)` and so on. To do so, just define those extra signatures as needed:

```julia
minus(a::AbstractArray, b::TrackedArray) = Tracker.track(minus, a, b)
minus(a::TrackedArray, b::AbstractArray) = Tracker.track(minus, a, b)
```

## Tracked Internals

All `Tracked*` objects (`TrackedArray`, `TrackedReal`) are light wrappers around the `Tracked` type, which you can access via the `.tracker` field.

```julia
julia> x.tracker
Flux.Tracker.Tracked{Array{Float64,1}}(0x00000000, Flux.Tracker.Call{Nothing,Tuple{}}(nothing, ()), true, [5.0, 6.0], [-2.0, -2.0])
```

The `Tracker` stores the gradient of a given object, which we've seen before.

```julia
julia> x.tracker.grad
2-element Array{Float64,1}:
 -2.0
 -2.0
```

The tracker also contains a `Call` object, which simply represents a function call that was made at some point during the forward pass. For example, the `+` call would look like this:

```julia
julia> Tracker.Call(+, 1, 2)
Flux.Tracker.Call{Base.#+,Tuple{Int64,Int64}}(+, (1, 2))
```

In the case of the `y` we produced above, we can see that it stores the call that produced it -- that is, `W*x`.

```julia
julia> y.tracker.f
Flux.Tracker.Call{...}(*, (param([1.0 2.0; 3.0 4.0]), param([5.0, 6.0])))
```

Notice that because the arguments to the call may also be tracked arrays, storing their own calls, this means that `Tracker` ends up forming a data structure that records everything that happened during the forward pass (often known as a *tape*).

When we call `back!(y, [1, -1])`, the sensitivities `[1, -1]` simply get forwarded to `y`'s call (`*`), effectively calling

```julia
Tracker.back(*, [1, -1], W, x)
```

which in turn calculates the sensitivities of the arguments (`W` and `x`) and back-propagates through their calls. This is recursive, so it will walk the entire program graph and propagate gradients to the original model parameters.
