
"""
    gradient(f, args...)

Returns a tuple containing `∂f/∂x` for each argument `x`,
the derivative (for scalar `x`) or the gradient.
If no gradient is defined, `∂f/∂x` will be `nothing`.

`f(args...)` must be a real number, see [`Zygote.jacobian`](@ref) for array output.

By default, `Flux.gradient` calls Zygote. If you load Enzyme, then other methods become available.

See also [`withgradient`](@ref) to keep the value `f(args...)`.

# Examples

```jldoctest; setup=:(using Zygote)
julia> gradient(*, 2.0, 3.0, 5.0)
(15.0, 10.0, 6.0)

julia> gradient(x -> sum(abs2,x), [7.0, 11.0, 13.0])
([14.0, 22.0, 26.0],)

julia> gradient([7, 11], 0, 1) do x, y, d
         p = size(x, d)
         sum(x.^p .+ y)
       end
([14.0, 22.0], 2.0, nothing)
```
"""
gradient(f, args...) = Zygote.gradient(f, args...)



"""
    withgradient(f, args...)

Returns both the value of the function and the [`gradient`](@ref), as a named tuple.

By default, `Flux.withgradient` calls Zygote. If you load Enzyme, then other methods become available.

# Example

```jldoctest; setup=:(using Zygote)
julia> y, ∇ = withgradient(/, 1, 2)
(val = 0.5, grad = (0.5, -0.25))

julia> ∇ == gradient(/, 1, 2)
true
```

Allows you to capture auxillary outputs, in addition to the scalar
used by `gradient`. To do this, `f` must return a Tuple or NamedTuple.
Then it calculates `grad = gradient(first∘f, args...)
but returns the whole `val = f(args...)`:

```jldoctest; setup=:(using Zygote)
julia> withgradient([1,2,4]) do x
          z = 1 ./ x
          sum(z), z  # here z is an auxillary output
       end
(val = (1.75, [1.0, 0.5, 0.25]), grad = ([-1.0, -0.25, -0.0625],))

julia> withgradient(3.0, 4.0) do x, y
          (div = x/y, mul = x*y)
       end
(val = (div = 0.75, mul = 12.0), grad = (0.25, -0.1875))
```
"""
withgradient(f, args...) = Zygote.withgradient(f, args...)
