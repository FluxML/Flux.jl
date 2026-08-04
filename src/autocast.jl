
# The scope stores the concrete `Type{Float16}`/`Type{BFloat16}` (or `nothing`) so that a
# read is a small concrete union: dispatching the per-layer barrier on it specializes each
# branch on a single precision, keeping the forward pass out of `Any`-typed territory.
const AUTOCAST_ELTYPE = ScopedValue{Union{Nothing, Type{Float16}, Type{BFloat16}}}(nothing)

# Autocast is compiled out entirely until its first use. `autocast_active()` is a
# constant-`false` method that the compiler folds away, so the half-precision branches
# below are dead-stripped from every layer's forward pass: outside of autocast, layers
# infer their exact concrete return type, pay zero overhead, and contain no half-precision
# types in their IR. The first `autocast` call in a session redefines the method to return
# `true` — a one-time world-age flip that invalidates and recompiles the affected layer
# code with the scope checks included (from then on forward passes infer as the small
# union of the three precision paths).
autocast_active() = false

const AUTOCAST_FLIP_LOCK = ReentrantLock()

function _ensure_autocast_active()
    autocast_active() && return nothing
    lock(AUTOCAST_FLIP_LOCK) do
        # re-check under the lock in the latest world: another task may have flipped,
        # and this specialization was compiled before the flip so it folds `false`
        if !Base.invokelatest(autocast_active)
            @eval autocast_active() = true
        end
    end
    return nothing
end

"""
    autocast_eltype()

Return the floating point type of the innermost enclosing [`autocast`](@ref) scope,
or `nothing` when called outside any `autocast` scope.

Custom layers can use this, together with the (internal) cast helpers used by the
built-in layers, to opt into mixed precision:

```julia
function (m::MyLayer)(x)
    T = Flux.autocast_eltype()
    W = Flux._autocast_down(T, m.weight)
    xT = Flux._autocast_down(T, x)
    return W * xT
end
```
"""
autocast_eltype() = AUTOCAST_ELTYPE[]

ChainRulesCore.@non_differentiable autocast_eltype()
EnzymeCore.EnzymeRules.inactive(::typeof(autocast_eltype), args...) = true

"""
    autocast(f, T::Type)

Run `f()` with mixed precision: while inside `f`, the forward pass of matmul- and
convolution-heavy Flux layers (`Dense`, `Conv`, `ConvTranspose`, `CrossCor`, `Bilinear`,
`Embedding` on onehot input, `MultiHeadAttention`, and the recurrent cells) casts
parameters and inputs to the half-precision type `T` (`Float16` or `BFloat16`) before
computing, while numerically sensitive operations (the normalization layers and the
loss functions) compute in `Float32`.

The model's parameters are not modified: they act as `Float32` "master weights",
and the gradients returned by [`Flux.gradient`](@ref) and [`Flux.withgradient`](@ref)
are accumulated back in `Float32`, so the usual optimiser setup works unchanged.
This mirrors PyTorch's `torch.autocast` recipe for mixed-precision training,
at layer rather than operator granularity.

Usually used through the `autocast` keyword of [`Flux.gradient`](@ref),
[`Flux.withgradient`](@ref) and [`Flux.train!`](@ref) rather than directly.

Note that with `T = Float16` gradients can underflow; robust `Float16` training
typically also needs loss scaling, which Flux does not provide yet.
`BFloat16` has the same exponent range as `Float32` and does not need it.

Autocast is compiled out until its first use: code that never calls `autocast` pays no
overhead, and layer forward passes keep their exact inferred return types. The first
`autocast` call in a session enables the machinery globally, which triggers a one-time
recompilation of the affected layer code (from then on, forward passes infer as the
small union of the `Float32`/`Float16`/`BFloat16` paths).

See also [`f16`](@ref) and [`bf16`](@ref) for statically converting a model instead.

# Examples

```julia-repl
julia> model = Chain(Dense(3 => 4, relu), BatchNorm(4), Dense(4 => 2));

julia> x = randn(Float32, 3, 8);

julia> y = autocast(BFloat16) do
           model(x)
       end;

julia> eltype(y)  # the final Dense ran in BFloat16
BFloat16

julia> eltype(model[1].weight)  # parameters are untouched
Float32

julia> grad = Flux.gradient(m -> sum(abs2, m(x)), model; autocast=BFloat16)[1];

julia> eltype(grad.layers[1].weight)  # gradients are Float32, like the parameters
Float32
```
"""
function autocast(f, ::Type{T}) where {T<:Union{Float16, BFloat16}}
    _ensure_autocast_active()
    # `invokelatest`: this call may sit in a specialization compiled before the flip
    return Base.invokelatest(with, f, AUTOCAST_ELTYPE => T)
end

autocast(f, ::Type{T}) where T =
    throw(ArgumentError("autocast supports Float16 and BFloat16, got $T"))

_with_autocast(g, ::Nothing) = g()
_with_autocast(g, ::Type{T}) where T = autocast(g, T)

# Run `f(autocast_eltype())` as a dispatch barrier: the scope value is a concrete type (or
# `nothing`), so each branch of the small union specializes `f` on a single precision. This
# collapses what would otherwise be `Union`-of-`Union` inference (e.g. `W * x` with both cast)
# into a per-precision concrete computation. Until the first `autocast` call flips
# `autocast_active()`, the whole scope consultation folds away to `f(nothing)`.
@inline function _autocast_barrier(f::F) where {F}
    if autocast_active()
        return f(autocast_eltype())
    else
        return f(nothing)
    end
end

# Cast to the autocast eltype, for matmul/conv-family layers. `nothing` (no active scope) and
# non-float arrays (onehot/integer inputs, `false` bias, `Nil`, ...) pass through unchanged.
# BFloat16 goes through `_to_bf16` rather than a native `convert`/broadcast, which can hang
# LLVM codegen on some platforms (JuliaMath/BFloat16s.jl#107).
_autocast_down(::Nothing, x) = x
_autocast_down(::Type{Float16}, x::AbstractArray{Float16}) = x
_autocast_down(::Type{Float16}, x::AbstractArray{<:AbstractFloat}) = Float16.(x)
_autocast_down(::Type{BFloat16}, x::AbstractArray{BFloat16}) = x
_autocast_down(::Type{BFloat16}, x::AbstractArray{<:AbstractFloat}) = _to_bf16(x)
_autocast_down(::Type{<:Union{Float16, BFloat16}}, x) = x

_autocast_down_pullback(proj) = dx -> (NoTangent(), NoTangent(), proj(unthunk(dx)))
function ChainRulesCore.rrule(::typeof(_autocast_down), ::Type{T},
                              x::AbstractArray{<:AbstractFloat}) where T
    proj = ChainRulesCore.ProjectTo(x)  # widens the cotangent back to eltype(x)
    return _autocast_down(T, x), _autocast_down_pullback(proj)
end
function ChainRulesCore.rrule(::typeof(_autocast_down), ::Nothing, x)
    return x, dx -> (NoTangent(), NoTangent(), dx)
end

# Cast half-precision arrays up to Float32 inside an autocast scope, for
# numerically sensitive operations (normalization layers, losses). Outside any
# scope this is a no-op, so statically converted `f16`/`bf16` models are unaffected.
_autocast_up(x) = autocast_active() ? _autocast_up_scoped(x) : x
_autocast_up_scoped(x) = autocast_eltype() === nothing ? x : _cast_f32(x)

_cast_f32(x::AbstractArray{<:Union{Float16, BFloat16}}) = convert(AbstractArray{Float32}, x)
_cast_f32(x) = x

function ChainRulesCore.rrule(::typeof(_cast_f32), x::AbstractArray{Float16})
    proj = ChainRulesCore.ProjectTo(x)
    cast_f32_pullback(dx) = (NoTangent(), proj(unthunk(dx)))
    return _cast_f32(x), cast_f32_pullback
end

# For BFloat16 the cotangent must be truncated through `_to_bf16`, not ProjectTo,
# again because of JuliaMath/BFloat16s.jl#107.
function ChainRulesCore.rrule(::typeof(_cast_f32), x::AbstractArray{BFloat16})
    cast_f32_bf16_pullback(dx) = (NoTangent(), _to_bf16(unthunk(dx)))
    return _cast_f32(x), cast_f32_bf16_pullback
end
