function applyGradient(p::Param)
    p.x .-= p.Δ
    p.Δ .= 0
end

function descent(p::Param, η::Real)
  function ()
	p.Δ .= p.Δ .* η
	applyGradient(p)
  end
end

function momentum(p::Param, ρ::Real)
  mo = zeros(p.x)
  function ()
	p.Δ .= mo .= ρ .* mo .+ p.Δ
	applyGradient(p)
  end
end

function nesterov(p::Param, ρ::Real)
  mo = zeros(p.x)
  function ()
    mo  .= ρ .* mo .+ p.Δ
    p.Δ .= ρ .* mo .+ p.Δ
	applyGradient(p)
  end
end

function clip(p::Param, thresh::Real)
  function ()
	clamp!(p.Δ, -thresh, thresh)
	applyGradient(p)
  end
end

function weightdecay(p::Param, γ::Real)
  function ()
	p.Δ .+= γ .* p.x
	applyGradient(p)
  end
end

function invdecay(p::Param, γ::Real)
  n = 0
  function ()
    p.Δ .*= 1 / (1 + γ * n)
    n += 1
	applyGradient(p)
  end
end

function rmsprop(p::Param; η::Real = 0.001, ρ::Real = 0.9, ϵ::Real = 1e-8)
  acc  = zeros(p.x) .+ ϵ
  function ()
    @. acc = ρ * acc + (1 - ρ) * p.Δ ^ 2
    @. p.Δ /= √acc * η
	applyGradient(p)
  end
end

function adagrad(p::Param; η::Real = 0.01, ϵ::Real = 1e-8)
  acc = zeros(p.x) .+ ϵ
  function ()
    @. acc += p.Δ ^ 2
    @. p.Δ /= √acc * η
	applyGradient(p)
  end
end

function adadelta(p::Param; ρ::Real = 0.95, ϵ::Real = 1e-8)
  acc = zeros(p.x) .+ ϵ
  Δacc = zeros(p.x) .+ ϵ
  function ()
    @. acc = ρ * acc + (1 - ρ) * p.Δ ^ 2
    @. p.Δ *= √Δacc / √acc
    @. Δacc = ρ * Δacc + (1 - ρ) * p.Δ ^ 2
	applyGradient(p)
  end
end

function adam(p::Param; η::Real = 0.001, β1::Real = 0.9, β2::Real = 0.999, ϵ::Real = 1e-8)
  mt = zeros(p.x)
  vt = zeros(p.x) .+ ϵ
  β1p, β2p = β1, β2
  function ()
    @. mt = β1 * mt + (1 - β1) * p.Δ
    @. vt = β2 * vt + (1 - β2) * p.Δ ^ 2
    @. p.Δ = √(1 - β2p) / √(1 - β1p) * mt / √vt * η
    β1p *= β1
    β2p *= β2
	applyGradient(p)
  end
end
