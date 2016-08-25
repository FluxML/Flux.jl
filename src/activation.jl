export σ, relu, softmax

σ(x) = 1 ./ (1 .+ exp.(-x))

back!(::typeof(σ), Δ, x) = Δ .* σ(x)./(1.-σ(x))

relu(x) = max(0, x)

back(::typeof(relu), Δ, x) = Δ .* (x .< 0)

softmax(x) = error("not implemented")
