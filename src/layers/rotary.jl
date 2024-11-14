
"""
    sincos_position_embed(pos, idx, hidden_size)

Calculate sinusoidal position embedding for a single position and feature index.
Uses alternating sine and cosine based on feature index parity.
"""
function sincos_position_embed(pos, idx, hidden_size)
    i = (idx + 1) ÷ 2  # which frequency we're calculating
    pos_idx = pos - 1

    # freq = 1/10000^(2i/d) following original transformer paper
    freq = 10.0f0^(-(8i/hidden_size))
    angle = pos_idx * freq

    # Alternate between sin and cos based on feature index
    return iseven(idx) ? cos(angle) : sin(angle)
end

ChainRulesCore.@non_differentiable sincos_position_embed(pos, idx, hidden_size)

"""
    _rotary_transform(x1, x2, cos_θ, sin_θ)

Helper function to perform rotary transformation of a feature pair.
"""
function _rotary_transform(x1, x2, cos_θ, sin_θ)
    y1 = x1 * cos_θ - x2 * sin_θ
    y2 = x2 * cos_θ + x1 * sin_θ
    return y1, y2
end

"""
    _apply_rotary(x, seq_len)

Apply rotary transformation to input tensor.
"""
function _apply_rotary(x, seq_len)
    hidden_size = size(x, 1)

    # Get positional encodings
    pos_enc = similar(x, hidden_size, seq_len)
    for i in 1:hidden_size, j in 1:seq_len
        pos_enc[i,j] = sincos_position_embed(j, i, hidden_size)
    end

    y = similar(x)
    half_size = hidden_size ÷ 2

    # Apply rotary transformation
    for j in 1:seq_len
        for i in 1:half_size
            idx = 2i - 1
            x1, x2 = x[idx,j], x[idx+1,j]
            cos_θ, sin_θ = pos_enc[idx,j], pos_enc[idx+1,j]

            y[idx,j], y[idx+1,j] = _rotary_transform(x1, x2, cos_θ, sin_θ)
        end
    end

    return y
end

"""
    with_rotary_position_embedding(x)

Apply rotary position embeddings to input tensor x.

Input tensor should be of shape (features, sequence_length, ...).
Features dimension must be even.

# Arguments
- `x`: Input tensor of shape (features, sequence_length, ...)

# Returns
- Tensor of same shape as input with rotary position embeddings applied
"""
function with_rotary_position_embedding(x::AbstractArray)
    hidden_size = size(x, 1)
    iseven(hidden_size) || throw(ArgumentError("Feature dimension ($(hidden_size)) must be even"))
    return _apply_rotary(x, size(x, 2))
end

# Gradient rules
function ChainRulesCore.rrule(::typeof(_rotary_transform), x1, x2, cos_θ, sin_θ)
    y1, y2 = _rotary_transform(x1, x2, cos_θ, sin_θ)

    function rotary_transform_pullback(Ȳ)
        ∂y1, ∂y2 = Ȳ

        # Inverse rotation matrix for gradients
        ∂x1 = ∂y1 * cos_θ + ∂y2 * sin_θ
        ∂x2 = -∂y1 * sin_θ + ∂y2 * cos_θ

        return (NoTangent(), ∂x1, ∂x2, NoTangent(), NoTangent())
    end

    return (y1, y2), rotary_transform_pullback
end

function ChainRulesCore.rrule(::typeof(_apply_rotary), x, seq_len)
    y = _apply_rotary(x, seq_len)

    function apply_rotary_pullback(Ȳ)
        hidden_size = size(x, 1)
        pos_enc = similar(x, hidden_size, seq_len)
        for i in 1:hidden_size, j in 1:seq_len
            pos_enc[i,j] = sincos_position_embed(j, i, hidden_size)
        end

        ∂x = similar(Ȳ)
        half_size = hidden_size ÷ 2

        for j in 1:seq_len
            for i in 1:half_size
                idx = 2i - 1
                cos_θ, sin_θ = pos_enc[idx,j], pos_enc[idx+1,j]
                ∂y1, ∂y2 = Ȳ[idx,j], Ȳ[idx+1,j]

                ∂x[idx,j] = ∂y1 * cos_θ + ∂y2 * sin_θ
                ∂x[idx+1,j] = -∂y1 * sin_θ + ∂y2 * cos_θ
            end
        end

        return (NoTangent(), ∂x, NoTangent())
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
