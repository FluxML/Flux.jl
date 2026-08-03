
# Mixed precision via layer wrappers (PyTorch `torch.autocast` style, at layer granularity).
#
# `autocast(model, T)` walks the model and wraps each matmul/convolution-heavy layer in an
# `AutocastDown{T}` (casts its parameters and inputs to the half-precision type `T` before
# computing) and each normalization layer in an `AutocastUp` (casts its input up to `Float32`).
# The wrapped layers run the *original* layer code, so the model's parameters are never
# modified — they act as `Float32` master weights, and gradients come back in `Float32`.
#
# Because the wrapping is a plain (differentiable) functor transform, running it inside a
# gradient closure routes gradients back to the original model's structure, and the wrapped
# forward passes keep their exact inferred element type (no runtime scope, no type widening).

# --- per-layer-type policy ---------------------------------------------------------------

"""
    Flux.autocast_mode(layer) -> Symbol

Trait controlling how [`autocast`](@ref) treats a layer. Returns `:down` for layers whose
parameters and inputs should be cast to the half-precision type (matmul/convolution family),
`:up` for layers that should compute in `Float32` (normalization), or `:none` (default) for
layers that `autocast` recurses into rather than wrapping.

Overload it to make a custom layer autocast-aware, e.g.
`Flux.autocast_mode(::MyLinear) = :down`.
"""
autocast_mode(x) = :none

# --- wrappers ----------------------------------------------------------------------------

"""
    AutocastDown{T}(layer)

Wrapper produced by [`autocast`](@ref) around a compute-heavy layer: on each call it casts
the layer's floating-point parameters and inputs to the half-precision type `T` before
running `layer`. Not usually constructed directly.
"""
struct AutocastDown{T, L}
    layer::L
end
AutocastDown{T}(layer::L) where {T, L} = AutocastDown{T, L}(layer)

"""
    AutocastUp(layer)

Wrapper produced by [`autocast`](@ref) around a numerically sensitive layer (normalization):
on each call it casts half-precision inputs up to `Float32` so the layer computes in full
precision. Not usually constructed directly.
"""
struct AutocastUp{L}
    layer::L
end

@layer :expand AutocastDown
@layer :expand AutocastUp

# Recurrent cells reach their initial state through `initialstates`; forward it so a wrapped
# cell can still be dropped into `RNN`/`LSTM`/`GRU`/`Recurrence`.
initialstates(w::AutocastDown) = initialstates(w.layer)
initialstates(w::AutocastUp) = initialstates(w.layer)

function (w::AutocastDown{T})(xs...) where T
    fields, re = Functors.functor(w.layer)
    layer_T = re(map(f -> _autocast_down(T, f), fields))
    return layer_T(map(x -> _autocast_down(T, x), xs)...)
end

(w::AutocastUp)(xs...) = w.layer(map(_autocast_up, xs)...)

# --- the transform -----------------------------------------------------------------------

"""
    autocast(model, T::Type)

Wrap `model` for mixed-precision execution with the half-precision type `T` (`Float16` or
`BFloat16`), returning a new model that shares `model`'s parameter arrays. In the wrapped
model, matmul- and convolution-heavy layers (`Dense`, `Conv`, `ConvTranspose`, `CrossCor`,
`Bilinear`, `Embedding`, `MultiHeadAttention`'s projections, and the recurrent cells) cast
their parameters and inputs to `T` before computing, while the normalization layers compute
in `Float32`.

The parameters themselves stay `Float32` ("master weights"): the wrappers cast on the fly,
so `model`'s arrays are untouched and gradients are accumulated back in `Float32`. This
mirrors PyTorch's `torch.autocast`, at layer rather than operator granularity, and — unlike
casting the parameters with [`f16`](@ref)/[`bf16`](@ref) — keeps forward passes type-stable.

Usually applied through the `autocast` keyword of [`Flux.gradient`](@ref),
[`Flux.withgradient`](@ref) and [`Flux.train!`](@ref) rather than directly; use the
two-argument form for a wrapped model to run at inference time.

Custom layers can opt in via [`Flux.autocast_mode`](@ref).

# Examples

```julia-repl
julia> model = Chain(Dense(3 => 4, relu), BatchNorm(4), Dense(4 => 2));

julia> x = randn(Float32, 3, 8);

julia> mac = autocast(model, BFloat16);

julia> eltype(mac(x))  # the final Dense ran in BFloat16
BFloat16

julia> eltype(model[1].weight)  # parameters are untouched
Float32

julia> grad = Flux.gradient(m -> sum(abs2, m(x)), model; autocast=BFloat16)[1];

julia> eltype(grad.layers[1].weight)  # gradients are Float32, like the parameters
Float32
```
"""
function autocast(model, ::Type{T}) where {T<:Union{Float16, BFloat16}}
    return fmap(model; exclude = _autocast_isleaf) do x
        mode = autocast_mode(x)
        mode === :down ? AutocastDown{T}(x) :
        mode === :up   ? AutocastUp(x)       : x
    end
end

autocast(model, ::Type{T}) where T =
    throw(ArgumentError("autocast supports Float16 and BFloat16, got $T"))

_autocast_isleaf(x) = autocast_mode(x) !== :none || Functors.isleaf(x)

# Thread `autocast=T` through `gradient`/`withgradient`/`train!`: return a closure that wraps
# every argument (a no-op on data arrays and other non-layer args) and calls `f`. Applied
# INSIDE the differentiated region, so the pullback of the wrapper construction maps gradients
# back to the original model's structure. `nothing` returns `f` unchanged (zero overhead).
_autocast_closure(f, ::Nothing) = f
_autocast_closure(f::F, ::Type{T}) where {F, T} = (args...) -> f(_map_autocast(T, args)...)

# Recursive tuple construction rather than `map`: Zygote differentiates this cleanly, whereas
# its adjoint for `map` over a heterogeneous tuple mishandles the tangent.
_map_autocast(::Type, ::Tuple{}) = ()
_map_autocast(::Type{T}, args::Tuple) where T =
    (autocast(first(args), T), _map_autocast(T, Base.tail(args))...)

# --- cast helpers ------------------------------------------------------------------------

# Down-cast for the wrapped compute layers. Non-float leaves (a `false` bias, integer/onehot
# inputs, `Nil`, activation functions, ...) pass through; tuples (e.g. a recurrent `(h, c)`
# state) are cast element-wise. BFloat16 goes through `_to_bf16` rather than a native
# `convert`/broadcast, which can hang LLVM codegen on some platforms (JuliaMath/BFloat16s.jl#107).
_autocast_down(::Type{Float16}, x::AbstractArray{Float16}) = x
_autocast_down(::Type{Float16}, x::AbstractArray{<:AbstractFloat}) = Float16.(x)
_autocast_down(::Type{BFloat16}, x::AbstractArray{BFloat16}) = x
_autocast_down(::Type{BFloat16}, x::AbstractArray{<:AbstractFloat}) = _to_bf16(x)
_autocast_down(::Type{T}, t::Tuple) where T = map(x -> _autocast_down(T, x), t)
_autocast_down(::Type, x) = x

_autocast_down_pullback(proj) = dx -> (NoTangent(), NoTangent(), proj(unthunk(dx)))
function ChainRulesCore.rrule(::typeof(_autocast_down), ::Type{T},
                              x::AbstractArray{<:AbstractFloat}) where T
    proj = ChainRulesCore.ProjectTo(x)  # widens the cotangent back to eltype(x)
    return _autocast_down(T, x), _autocast_down_pullback(proj)
end

# Up-cast half-precision arrays to Float32 for the wrapped normalization layers (and, always,
# for the loss functions — see `_upcast_half`). Widening, so CPU-safe for BFloat16.
_autocast_up(x::AbstractArray{<:Union{Float16, BFloat16}}) = convert(AbstractArray{Float32}, x)
_autocast_up(x) = x

function ChainRulesCore.rrule(::typeof(_autocast_up), x::AbstractArray{Float16})
    proj = ChainRulesCore.ProjectTo(x)
    return _autocast_up(x), dx -> (NoTangent(), proj(unthunk(dx)))
end
# For BFloat16 the cotangent must be truncated through `_to_bf16`, not `ProjectTo`, again
# because of JuliaMath/BFloat16s.jl#107.
function ChainRulesCore.rrule(::typeof(_autocast_up), x::AbstractArray{BFloat16})
    return _autocast_up(x), dx -> (NoTangent(), _to_bf16(unthunk(dx)))
end

# Unconditional Float32 accumulation for loss functions given half-precision inputs. Unlike
# the norm up-cast this is not gated on a wrapper: a loss is called by the user on the model
# output, so there is no wrapper to carry the policy. Integer labels and onehot targets pass
# through untouched.
_upcast_half(x) = _autocast_up(x)

# --- per-layer policy for the built-in layers --------------------------------------------

# Matmul/convolution-heavy layers: cast parameters + inputs to the half-precision type.
autocast_mode(::Dense) = :down
autocast_mode(::Bilinear) = :down
autocast_mode(::Conv) = :down
autocast_mode(::ConvTranspose) = :down
autocast_mode(::CrossCor) = :down
autocast_mode(::RNNCell) = :down
autocast_mode(::LSTMCell) = :down
autocast_mode(::GRUCell) = :down
autocast_mode(::GRUv3Cell) = :down

# Normalization: compute in Float32.
autocast_mode(::BatchNorm) = :up
autocast_mode(::InstanceNorm) = :up
autocast_mode(::GroupNorm) = :up
autocast_mode(::LayerNorm) = :up

# `MultiHeadAttention` is left to recurse: its `q_proj`/`k_proj`/`v_proj`/`out_proj` are
# `Dense` layers that get wrapped individually, so the attention (and its softmax) runs in
# the half-precision type. `Embedding` is deliberately NOT wrapped: casting the (often large)
# embedding table on every forward would be expensive, and PyTorch keeps embeddings in full
# precision under autocast — downstream wrapped layers cast the looked-up vectors instead.
