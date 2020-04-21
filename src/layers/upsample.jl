struct BilinearUpsample{T<:Integer}
    factor::Tuple{T,T}
end

function (b::BilinearUpsample)(x::AbstractArray)
    W, H, C, N = size(x)

    newW = W * b.factor[1]
    newH = H * b.factor[2]

    out = similar(x, (newW, newH, C, N))

    for n = 1:N, c = 1:C, w = 1:newW, h = 1:newH
        w₀ = (w - 0.5) / b.factor[1] + 0.5
        h₀ = (h - 0.5) / b.factor[2] + 0.5

        w1 = floor(Int, w₀)
        w2 = w1 + 1
        h1 = floor(Int, h₀)
        h2 = h1 + 1

        i1 = clamp(w1, 1, W)
        i2 = clamp(w2, 1, W)
        j1 = clamp(h1, 1, H)
        j2 = clamp(h2, 1, H)

        out[w, h, c, n] =
            (
                x[i1, j1, c, n] * (w2 - w₀) * (h2 - h₀) +
                x[i1, j2, c, n] * (w2 - w₀) * (h₀ - h1) +
                x[i2, j1, c, n] * (w₀ - w1) * (h2 - h₀) +
                x[i2, j2, c, n] * (w₀ - w1) * (h₀ - h1)
            ) / (w2 - w1 * h2 - h1)
    end

    out
end

function Base.show(io::IO, b::BilinearUpsample)
    print(io, "BilinearUpsample(", b.factor[1], ", ", b.factor[2], ")")
end
