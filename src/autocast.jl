
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

Trait controlling how [`autocast`](@ref) treats a layer. Returns one of:

  - `:down` — cast the layer's parameters *and* inputs to the half-precision type
    (matmul/convolution family);
  - `:up` — cast the layer's inputs up to `Float32` so it computes in full precision
    (numerically sensitive normalization such as `LayerNorm`);
  - `:keep` — cast the layer's inputs to the half-precision type but leave its parameters in
    `Float32`, so the activation stays half while the kernel keeps its statistics in
    `Float32` (`BatchNorm`);
  - `:none` (default) — do not wrap; `autocast` recurses into the layer instead.

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

"""
    AutocastKeep{T}(layer)

Wrapper produced by [`autocast`](@ref) around a statistics-based normalization layer
(`BatchNorm`): casts the layer's *inputs* to the half-precision type `T` but leaves its
parameters (scale/shift and running statistics) in `Float32`. The layer therefore computes
with half-precision activations and `Float32` statistics — exactly as [`bf16mix`](@ref)/
[`f16mix`](@ref) do, and as cuDNN's batch-norm kernel does natively — so the activation
stays in `T` through the layer instead of taking the full-precision round-trip that
`AutocastUp` incurs (an up-cast in, and a down-cast by the next layer out). Not usually
constructed directly.
"""
struct AutocastKeep{T, L}
    layer::L
end

AutocastKeep{T}(layer::L) where {T, L} = AutocastKeep{T, L}(layer)

@layer :expand AutocastDown
@layer :expand AutocastUp
@layer :expand AutocastKeep

# Recurrent cells reach their initial state through `initialstates`; forward it so a wrapped
# cell can still be dropped into `RNN`/`LSTM`/`GRU`/`Recurrence`.
initialstates(w::AutocastDown) = initialstates(w.layer)
initialstates(w::AutocastUp) = initialstates(w.layer)
initialstates(w::AutocastKeep) = initialstates(w.layer)

function (w::AutocastDown{T})(xs...) where T
    fields, re = Functors.functor(w.layer)
    layer_T = re(map(f -> _autocast_down(T, f), fields))
    return layer_T(map(x -> _autocast_down(T, x), xs)...)
end

(w::AutocastUp)(xs...) = w.layer(map(_autocast_up, xs)...)

# Run the layer with its Float32 parameters untouched. On the GPU the activation is kept in
# `T` (its kernel — cuDNN for `BatchNorm` — folds the statistics in Float32 internally), so it
# flows through in half precision with no round-trip. On the CPU there is no such fast kernel:
# half-precision norms are software-emulated (slower, not faster) and `BFloat16` arithmetic
# additionally miscompiles on Julia ≥1.11/x86 (JuliaMath/BFloat16s.jl#107), so the input is
# cast *up* to Float32 there — the same as `AutocastUp`, since the CPU gains nothing from half.
(w::AutocastKeep{T})(xs...) where T = w.layer(map(x -> _keep_cast(T, x), xs)...)

_keep_cast(::Type{T}, x::Array) where T = _autocast_up(x)              # CPU: compute in Float32
_keep_cast(::Type{T}, x::AbstractArray) where T = _autocast_down(T, x)  # GPU: keep half precision
_keep_cast(::Type, x) = x

# --- the transform -----------------------------------------------------------------------

"""
    autocast(model, T::Type)

Wrap `model` for mixed-precision execution with the half-precision type `T` (`Float16` or
`BFloat16`), returning a new model that shares `model`'s parameter arrays. In the wrapped
model, matmul- and convolution-heavy layers (`Dense`, `Conv`, `ConvTranspose`, `CrossCor`,
`Bilinear`, `Embedding`, `MultiHeadAttention`'s projections, and the recurrent cells) cast
their parameters and inputs to `T` before computing. `BatchNorm` keeps its parameters in
`Float32` but lets the half-precision activation pass through (its statistics are folded in
`Float32` inside the kernel); the other normalization layers cast their input up to
`Float32`.

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
        mode === :up   ? AutocastUp(x)       :
        mode === :keep ? AutocastKeep{T}(x)  : x
    end
end

autocast(model, ::Type{T}) where T =
    throw(ArgumentError("autocast supports Float16 and BFloat16, got $T"))

_autocast_isleaf(x) = autocast_mode(x) !== :none || Functors.isleaf(x)

# Thread `autocast=T` through `gradient`/`withgradient`/`train!`: return a closure that wraps
# every argument (a no-op on data arrays and other non-layer args) and calls `f`. Applied
# INSIDE the differentiated region; the `rrule` for `autocast` below maps the wrapped-model
# gradient back to the original model's structure. `nothing` returns `f` unchanged (zero cost).
_autocast_closure(f, ::Nothing) = f
_autocast_closure(f::F, ::Type{T}) where {F, T} = (args...) -> f(_map_autocast(T, args)...)

# Recursive tuple construction rather than `map`: Zygote differentiates this cleanly, whereas
# its adjoint for `map` over a heterogeneous tuple mishandles the tangent.
_map_autocast(::Type, ::Tuple{}) = ()
_map_autocast(::Type{T}, args::Tuple) where T =
    (autocast(first(args), T), _map_autocast(T, Base.tail(args))...)

# Differentiating `autocast` by letting Zygote trace the `fmap` that builds ~one wrapper struct
# per layer is needlessly expensive (it re-derives the whole construction every step). Since
# the transform only *wraps* layers over shared parameter arrays, its adjoint is just the
# inverse — strip the wrappers from the cotangent — which we give directly as an `rrule`. This
# recovers the cost of differentiating a pre-wrapped model while keeping the wrap-inside-the-
# closure ergonomics (the returned gradient is shaped like the original, unwrapped model).
_iswrapper(x) = x isa AutocastDown || x isa AutocastUp || x isa AutocastKeep

# Pull child `k` (a field name or tuple index) out of a cotangent, tolerating the several forms
# an absent/zero tangent can take.
_child_tangent(::Nothing, k) = nothing
_child_tangent(Δ::ChainRulesCore.AbstractZero, k) = nothing
_child_tangent(Δ::NamedTuple, k::Symbol) = get(Δ, k, nothing)
_child_tangent(Δ::Tuple, k::Int) = Δ[k]
_child_tangent(Δ, k) = getproperty(Δ, k)   # ChainRulesCore.Tangent

# Reshape the cotangent `Δ` of the wrapped node `wm` to the *original* (unwrapped) structure,
# as NamedTuples/Tuples (Zygote's gradient convention). At a wrapper the inner `.layer`
# cotangent is already original-shaped (a wrapped layer contains no further wrappers), so we
# lift it out; otherwise we recurse through the unwrapped containers (`Chain`, `Parallel`, …).
_unwrap_grad(wm, ::Nothing) = nothing
_unwrap_grad(wm, Δ::ChainRulesCore.AbstractZero) = Δ
function _unwrap_grad(wm, Δ)
    _iswrapper(wm) && return _child_tangent(Δ, :layer)
    Functors.isleaf(wm) && return Δ
    xs, _ = Functors.functor(wm)
    return _map_child_grads(xs, Δ)
end
_map_child_grads(xs::NamedTuple{K}, Δ) where K =
    NamedTuple{K}(map(k -> _unwrap_grad(getfield(xs, k), _child_tangent(Δ, k)), K))
_map_child_grads(xs::Tuple, Δ) =
    ntuple(i -> _unwrap_grad(xs[i], _child_tangent(Δ, i)), length(xs))

function ChainRulesCore.rrule(::typeof(autocast), model,
                              ::Type{T}) where {T<:Union{Float16, BFloat16}}
    wm = autocast(model, T)
    autocast_pullback(Δ) = (NoTangent(), _unwrap_grad(wm, unthunk(Δ)), NoTangent())
    return wm, autocast_pullback
end

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

# Normalization. `BatchNorm` folds its statistics in `Float32` inside the kernel (cuDNN on the
# GPU) while the activation stays in the half type, so it keeps its parameters `Float32` but
# lets the half-precision activation pass through (`:keep`) — no full-precision round-trip.
# `LayerNorm`/`InstanceNorm`/`GroupNorm` reduce over the feature/spatial axes without a
# Float32-accumulating fast path, so they cast the activation up to `Float32` (`:up`).
autocast_mode(::BatchNorm) = :keep
autocast_mode(::InstanceNorm) = :up
autocast_mode(::GroupNorm) = :up
autocast_mode(::LayerNorm) = :up

# `MultiHeadAttention` is left to recurse: its `q_proj`/`k_proj`/`v_proj`/`out_proj` are
# `Dense` layers that get wrapped individually, so the attention (and its softmax) runs in
# the half-precision type. `Embedding` is deliberately NOT wrapped: casting the (often large)
# embedding table on every forward would be expensive, and PyTorch keeps embeddings in full
# precision under autocast — downstream wrapped layers cast the looked-up vectors instead.
