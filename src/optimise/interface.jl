call(f, xs...) = f(xs...)

function optimiser(ps, fs...)
  ps = [Param(p) for p in ps]
  fs = map(ps) do p
    os = map(f -> f(p), fs)
    () -> foreach(call, os)
  end
  () -> foreach(call, fs)
end

SGD(ps, η = 1) = optimiser(ps, p -> descent(p, η))
ADAM(ps, η = 0.001, β1 = 0.9, β2 = 0.999, ϵ = 1e-08, decay = 0.0) = optimiser(ps, p -> adam(p; η = η, β1 = β1, β2 = β2, ϵ = ϵ), p -> invdecay(p, decay), p -> descent(p, 1))
Momentum(ps,ρ, decay = 0.0) = optimiser(ps, p -> momentum(p, ρ), p -> invdecay(p, decay), p -> descent(p, 1))
Nesterov(ps,ρ, decay = 0.0) = optimiser(ps, p -> nesterov(p, ρ), p -> invdecay(p, decay), p -> descent(p, 1))
RMSProp(ps, η = 0.001, ρ = 0.9, ϵ = 1e-8, decay = 0.0) = optimiser(ps, p -> rmsprop(p; η = η, ρ = ρ, ϵ = ϵ), p -> invdecay(p, decay), p -> descent(p, 1))
ADAGrad(ps, η = 0.01, ϵ = 1e-8, decay = 0.0) = optimiser(ps, p -> adagrad(p; η = η, ϵ = ϵ), p -> invdecay(p, decay), p -> descent(p, 1))
ADADelta(ps, η = 0.01, ρ = 0.95, ϵ = 1e-8, decay = 0.0) = optimiser(ps, p -> adadelta(p; ρ = ρ, ϵ = ϵ), p -> invdecay(p, decay), p -> descent(p, 1))
