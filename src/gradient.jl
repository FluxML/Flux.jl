
"""
    gradient(f, args...)

Returns a tuple containing `∂f/∂x` for each argument `x`,
the derivative (for scalar `x`) or the gradient.
If no gradient is defined, `∂f/∂x` will be `nothing`.

`f(args...)` must be a real number, see [`Zygote.jacobian`](@ref) for array output.

By default, `Flux.gradient` calls Zygote. If you load Enzyme, then other methods become available.

See also [`withgradient`](@ref) to keep the value `f(args...)`.

# Examples

```
julia> Flux.gradient(*, 2.0, 3.0, 5.0)
(15.0, 10.0, 6.0)

julia> Flux.gradient(x -> sum(abs2,x), [7.0, 11.0, 13.0])
([14.0, 22.0, 26.0],)

julia> Flux.gradient([7, 11], 0, 1) do x, y, d
         p = size(x, d)
         sum(x.^p .+ y)
       end
([14.0, 22.0], 2.0, nothing)
```
"""
function gradient(f, args...; zero::Bool=true)
    for a in args
        a isa EnzymeCore.Duplicated && return _enzyme_gradient(f, map(_ensure_enzyme, args)...; zero)
    end
    for a in args
        a isa EnzymeCore.Const && throw(ArgumentError(
            "The method `gradient(f, xs...)` using Enzyme.jl requires at least one `Duplicated` argument, not just `Const`."
        ))
    end
    Zygote.gradient(f, args...)
end

_ensure_enzyme(x::EnzymeCore.Duplicated) = x
_ensure_enzyme(x::EnzymeCore.Const) = x
_ensure_enzyme(x) = EnzymeCore.Const(x)

"""
    gradient(f, args::Union{Const,Duplicated}...)

This should return the same answer as `gradient(f, args...)`,
but it uses Enzyme.jl instead of Zygote.jl to compute the derivative.

Only available when Enzyme is loaded!

This method is used when at least one argument is of type `Duplicated`,
and all unspecified aguments are wrapped in `Const`.

Besides returning the gradient, this is also stored within the `Duplicated` object.
Calling `Enzyme.Duplicated(model)` allocates space for the gradient,
which is zero'd befor use when calling `gradient`.
With the keyword `zero=false`, the new gradient will instead be added to what is already stored.

!!! warning "Experimental"
    Enzyme support like this is new and somewhat experimental.
    This method was added in Flux 0.15.

# Example
```
julia> using Flux

julia> model = Chain(Dense([3.0;;]));

julia> Flux.gradient(model, [1]) do m, x  # computed using Zygote
         sum(abs2, m(x))
       end
((layers = ((weight = [6.0;;], bias = [6.0], σ = nothing),),), [18.0])

julia> using Enzyme

julia> dup_model = Duplicated(model);  # allocates space for gradient

julia> Flux.gradient(dup_model, Const([1])) do m, x  # Enzyme, returns the same
         sum(abs2, m(x))
       end
((layers = ((weight = [6.0;;], bias = [6.0], σ = nothing),),), nothing)

julia> dup_model  # same gradient is also stored within Duplicated
Duplicated(
  Chain(
    Dense(1 => 1),                      # 2 parameters
  ),
  # norm(∇) ≈ 8.49
)

julia> Flux.destructure((weight = [6.0;;], bias = [6.0]))[1] |> norm
8.48528137423857

julia> Flux.gradient(dup_model, [1]; zero=false) do m, x  # implict Const([1]), and grad accumulation
         sum(abs2, m(x))
       end
((layers = ((weight = [12.0;;], bias = [12.0], σ = nothing),),), nothing)
```
"""
gradient(f, args::Union{EnzymeCore.Const, EnzymeCore.Duplicated}...; zero::Bool=true) = _enzyme_gradient(f, args...; zero)

gradient(f, args::EnzymeCore.Const...; zero::Bool=true) = throw(ArgumentError(
    "The method `gradient(f, xs...)` using Enzyme.jl requires at least one `Duplicated` argument, not just `Const`."
))

# FluxEnzymeExt defines more specific _enzyme_gradient(f, args::Union{Const, Duplicated}...; zero)
_enzyme_gradient(f, args...; zero) = throw(ArgumentError(
    "Methods like `gradient(f, x::Duplicated)` are only available when Enzyme is loaded."
))


"""
    withgradient(f, args...)

Returns both the value of the function and the [`gradient`](@ref), as a named tuple.

By default, `Flux.withgradient` calls Zygote. If you load Enzyme, then other methods become available.

# Example

```
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
function withgradient(f, args...; zero::Bool=true)
    for a in args
        a isa EnzymeCore.Duplicated && return _enzyme_withgradient(f, map(_ensure_enzyme, args)...; zero)
    end
    for a in args
        a isa EnzymeCore.Const && throw(ArgumentError(
            "The method `withgradient(f, xs...)` using Enzyme.jl requires at least one `Duplicated` argument, not just `Const`."
        ))
    end
    Zygote.withgradient(f, args...)
end

"""
    withgradient(f, args::Union{Const,Duplicated}...)

This should return the same answer as `withgradient(f, model, args...)`,
but it uses Enzyme.jl instead of Zygote.jl to compute the derivative.

Only available when Enzyme is loaded!

!!! warning "Experimental"
    Enzyme support like this is new and somewhat experimental.
    This method was added in Flux 0.15.

# Example

```julia
julia> using Flux, Enzyme

julia> model = Chain(Embedding([1.1 2.2 3.3]), Dense([4.4;;]), only);

julia> model(3)
14.52

julia> Flux.withgradient(m -> m(3), model)  # this uses Zygote
(val = 14.52, grad = ((layers = ((weight = [0.0 0.0 4.4],), (weight = [3.3;;], bias = [1.0], σ = nothing), nothing),),))

julia> Flux.withgradient(m -> m(3), Duplicated(model))  # this uses Enzyme
(val = 14.52, grad = ((layers = ((weight = [0.0 0.0 4.4],), (weight = [3.3;;], bias = [1.0], σ = nothing), nothing),),))
```

The function `f` may return Tuple or NamedTuple, with the loss as the first element.
The gradient is then `grad = gradient(first∘f, args...)`
but the returned value is `val = f(args...)`:

```julia
julia> Flux.withgradient(m -> (m(3), "aux"), Duplicated(model))
(val = (14.52, "aux"), grad = ((layers = ((weight = [0.0 0.0 4.4],), (weight = [3.3;;], bias = [1.0], σ = nothing), nothing),),))

julia> Flux.withgradient(m -> (loss=m(3), aux=round.(m.(1:3); digits=3)), Duplicated(model))
(val = (loss = 14.52, aux = [4.84, 9.68, 14.52]), grad = ((layers = ((weight = [0.0 0.0 4.4],), (weight = [3.3;;], bias = [1.0], σ = nothing), nothing),),))
```
"""
withgradient(f, args::Union{EnzymeCore.Const, EnzymeCore.Duplicated}...; zero::Bool=true) = _enzyme_withgradient(f, args...; zero)

withgradient(f, args::EnzymeCore.Const...; zero::Bool=true) = throw(ArgumentError(
    "The method `withgradient(f, xs...)` using Enzyme.jl requires at least one `Duplicated` argument, not just `Const`."
))

# FluxEnzymeExt defines more specific _enzyme_withgradient(f, args::Union{Const, Duplicated}...; zero)
_enzyme_withgradient(f, args...; zero) = throw(ArgumentError(
    "Methods like `withgradient(f, x::Duplicated)` are only available when Enzyme is loaded."
))
