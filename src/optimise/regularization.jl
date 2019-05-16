# Proximal updates for convex regularization
using LinearAlgebra

struct L1_regularization
    α::Float64
    f::Function
end

shrink(α) = f(z) = z > α ? α : z < -α ? -α : z

function L1_regularization(α)
    return L1_regularization(α, shrink(α))
end

function apply!(r::L1_regularization, x, Δ)
    z = data(x)
    Δ .= r.f.(z)
    return Δ
end

struct L2_regularization
    α::Float64
end

function apply!(r::L2_regularization, x, Δ)
    z = data(x)
    norm_z = norm(z)
    if norm_z > r.α
        Δ .= (r.α/norm_z) .* z
    else
        Δ .= z
    end
    return Δ
end
