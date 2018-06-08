# Flux.Tracker

Backpropagation, or reverse-mode automatic differentiation, is handled by the `Flux.Tracker` module.

```julia
julia> using Flux.Tracker
```

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

## Internals

All `Tracked*` objects (`TrackedArray`, `TrackedReal`) are light wrappers around the `Tracked` type, which you can access via the `.tracker` field.

```julia
julia> x.tracker
Flux.Tracker.Tracked{Array{Float64,1}}(0x00000000, Flux.Tracker.Call{Void,Tuple{}}(nothing, ()), true, [5.0, 6.0], [-2.0, -2.0])
```

The `Tracker` stores the value and gradient of a given object, which we've seen before.

```julia
julia> x.tracker.data
2-element Array{Float64,1}:
 5.0
 6.0

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

which in turn calculates the sensitivities of the arguments (`W` and `x`) and backpropagates through their calls. This is recursive, so it will walk the entire program graph and propagate gradients to the original model parameters.

## Custom Gradients

We can hook in to the processes above to implement custom gradients for a function or kernel. For a toy example, imagine a custom implementation of `minus`:

```julia
julia> minus(a, b) = a - b
```

Firstly, we must tell the tracker system to stop when it sees a call to `minus`, and record it. We can do this using dispatch:

```julia
julia> minus(a::TrackedArray, b::TrackedArray) = Tracker.track(minus, a, b)
minus (generic function with 2 methods)
```

`Tracker.track` does two things: (1) it makes sure `minus` is called with *normal* array, not tracked ones (you can use `@show` inside `minus` to verify this), and (2) it uses the result to add a `minus` node to the tape. Look inside the result of calling `minus` to see what happened:

```julia
julia> a, b = param([6,5,4]), param([1,2,3])
(param([6.0, 5.0, 4.0]), param([1.0, 2.0, 3.0]))

julia> c = minus(a, b)
Tracked 3-element Array{Float64,1}:
 5.0
 3.0
 1.0

julia> c.tracker.f
Flux.Tracker.Call{...}(minus, (param([6.0, 5.0, 4.0]), param([1.0, 2.0, 3.0])))
```

Finally, we have to specify the gradient of `minus`.

```julia
julia> Tracker.back(::typeof(minus), Δ, a, b) =
        (Tracker.@back(a, Δ); Tracker.@back(b, -Δ))
```

`@back(x, Δ)` tells the tracker to continue propagating the sensitivity `Δ` through `x`. Now, AD will work with any program that calls `minus`.

```julia
julia> Flux.back!(c, 1)

julia> a.grad
3-element Array{Float64,1}:
 1.0
 1.0
 1.0

julia> b.grad
3-element Array{Float64,1}:
 -1.0
 -1.0
 -1.0
```

## Notes

For multi-argument functions with custom gradients, you likely want to catch not just `minus(::TrackedArray, ::TrackedArray)` but also `minus(::Array, TrackedArray)` and so on. To do so, just define those extra signatures as needed:

```julia
minus(a::AbstractArray, b::TrackedArray) = Tracker.track(minus, a, b)
minus(a::TrackedArray, b::AbstractArray) = Tracker.track(minus, a, b)
```

`@back` *must* be called exactly once on each tracked input argument. You do not need to do any special handling if one of the arguments is not tracked, as `@back` will just become a no-op.
