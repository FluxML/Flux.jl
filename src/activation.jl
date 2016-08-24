export σ

σ(x) = 1 ./ (1 .+ exp.(-x))

back!(::typeof(σ), Δ, x) = Δ .* σ(x)./(1.-σ(x))
