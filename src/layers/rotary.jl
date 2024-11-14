"""
    Rotary Position Embeddings (RoPE)

This is a port of the RoPE implementation from NeuralAttentionlib.jl, which is an implementation of
the Rotary Position Embeddings (RoPE) described in the RoFormer paper.

Original sources:
- Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  Authors: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen
  URL: https://arxiv.org/abs/2104.09864
  
- Code: NeuralAttentionlib.jl
  Author: chengchingwen
  Repository: https://github.com/chengchingwen/NeuralAttentionlib.jl

RoPE encodes absolute positional information with a rotation matrix that naturally 
incorporates explicit relative position dependency in self-attention formulation.
"""

"""
Calculate position-dependent frequency.
"""
function _default_freq_decay(i, hidden_size)
    j = 8 * (1 - i)
    return Float32(10^(j / hidden_size))
end

"""
Calculate sinusoidal position embedding.
"""
function sincos_position_embed(pos, idx, hidden_size)
    feature = Int32(idx)
    i = (feature + 1) >> 1  # integer divide by 2
    pos_idx = Int32(pos - 1)

    freq = _default_freq_decay(i, hidden_size)
    angle = pos_idx * freq

    return iseven(feature) ? cos(angle) : sin(angle)
end

ChainRulesCore.@non_differentiable sincos_position_embed(pos, idx, hidden_size)

"""
Apply rotation to a pair of values.
"""
function _rotary((x1, x2), (sin_θ, cos_θ))
    return (
        x1 * cos_θ - x2 * sin_θ,
        x2 * cos_θ + x1 * sin_θ
    )
end

"""
Apply rotary embeddings to the full tensor.
"""
function _apply_rotary(x, seq_len)
    hidden_size = size(x, 1)

    # Get positional encodings
    pos_enc = similar(x, hidden_size, seq_len)
    for i in 1:hidden_size, j in 1:seq_len
        pos_enc[i,j] = sincos_position_embed(j, i, hidden_size)
    end

    # Reshape to handle pairs properly
    x_reshaped = reshape(x, 2, :)
    pos_reshaped = reshape(pos_enc, 2, :)

    # Now reinterpret as pairs
    x_pairs = reinterpret(reshape, NTuple{2,eltype(x)}, x_reshaped)
    pos_pairs = reinterpret(reshape, NTuple{2,eltype(pos_enc)}, pos_reshaped)

    # Apply rotary transformation
    y_reshaped = similar(x_reshaped)
    y_pairs = reinterpret(reshape, NTuple{2,eltype(y_reshaped)}, y_reshaped)

    for i in axes(x_pairs, 1)
        y_pairs[i] = _rotary(x_pairs[i], pos_pairs[i])
    end

    # Reshape back to original dimensions
    return reshape(y_reshaped, size(x))
end

"""
Apply rotary position embeddings to input tensor x.
"""
function with_rotary_position_embedding(x::AbstractArray)
    hidden_size = size(x, 1)
    iseven(hidden_size) || throw(ArgumentError("Feature dimension ($(hidden_size)) must be even"))
    return _apply_rotary(x, size(x, 2))
end

# Gradient rules
function ChainRulesCore.rrule(::typeof(_rotary), x_pair, pos_pair)
    x1, x2 = x_pair
    sin_θ, cos_θ = pos_pair
    y1, y2 = _rotary(x_pair, pos_pair)

    function rotary_pullback(Ȳ)
        ∂y1, ∂y2 = Ȳ

        ∂x1 = ∂y1 * cos_θ + ∂y2 * sin_θ
        ∂x2 = -∂y1 * sin_θ + ∂y2 * cos_θ

        return (NoTangent(), (∂x1, ∂x2), NoTangent())
    end

    return (y1, y2), rotary_pullback
end

function ChainRulesCore.rrule(::typeof(_apply_rotary), x, seq_len)
    y = _apply_rotary(x, seq_len)

    function apply_rotary_pullback(Ȳ)
        hidden_size = size(x, 1)

        # Recalculate position encodings for gradient
        pos_enc = similar(x, hidden_size, seq_len)
        for i in 1:hidden_size, j in 1:seq_len
            pos_enc[i,j] = sincos_position_embed(j, i, hidden_size)
        end

        # Reshape for gradient computation
        x_reshaped = reshape(x, 2, :)
        pos_reshaped = reshape(pos_enc, 2, :)
        Ȳ_reshaped = reshape(Ȳ, 2, :)

        x_pairs = reinterpret(reshape, NTuple{2,eltype(x)}, x_reshaped)
        pos_pairs = reinterpret(reshape, NTuple{2,eltype(pos_enc)}, pos_reshaped)
        Ȳ_pairs = reinterpret(reshape, NTuple{2,eltype(Ȳ)}, Ȳ_reshaped)

        ∂x_reshaped = similar(x_reshaped)
        ∂x_pairs = reinterpret(reshape, NTuple{2,eltype(∂x_reshaped)}, ∂x_reshaped)

        for i in axes(x_pairs, 1)
            _, pb = rrule(_rotary, x_pairs[i], pos_pairs[i])
            ∂x_pairs[i] = pb(Ȳ_pairs[i])[2]
        end

        return (NoTangent(), reshape(∂x_reshaped, size(x)), NoTangent())
    end

    return y, apply_rotary_pullback
end

function ChainRulesCore.rrule(::typeof(with_rotary_position_embedding), x::AbstractArray)
    y = with_rotary_position_embedding(x)

    function rotary_pullback(Ȳ)
        _, ∂x, _ = rrule(_apply_rotary, x, size(x,2))[2](Ȳ)
        return (NoTangent(), ∂x)
    end

    return y, rotary_pullback
end
