call(f, xs...) = f(xs...)

function optimiser(ps, fs...)
  ps = [Param(p) for p in ps]
  fs = map(ps) do p
    os = map(f -> f(p), fs)
    () -> foreach(call, os)
  end
  () -> foreach(call, fs)
end

"""
    SGD(params, η = 1; decay = 0)

Classic gradient descent optimiser. For each parameter `p` and its
gradient `δp`, this runs `p -= η*δp`.

Supports decayed learning rate decay if the `decay` argument is provided.
"""
SGD(ps, η = 1; decay = 0) =
  optimiser(ps, p -> invdecay(p, decay), p -> descent(p, η))

"""
    Momentum(params, ρ, decay = 0)

SGD with momentum `ρ` and optional learning rate decay.
"""
Momentum(ps, ρ; decay = 0) =
  optimiser(ps, p -> momentum(p, ρ), p -> invdecay(p, decay), p -> descent(p, 1))

"""
    Nesterov(params, ρ, decay = 0)

SGD with Nesterov momentum `ρ` and optional learning rate decay.
"""
Nesterov(ps, ρ; decay = 0) =
  optimiser(ps, p -> nesterov(p, ρ), p -> invdecay(p, decay), p -> descent(p, 1))

"""
    RMSProp(params; η = 0.001, ρ = 0.9, ϵ = 1e-8, decay = 0)

[RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
optimiser. Parameters other than learning rate don't need tuning. Often a good
choice for recurrent networks.
"""
RMSProp(ps, η = 0.001; ρ = 0.9, ϵ = 1e-8, decay = 0) =
  optimiser(ps, p -> rmsprop(p; η = η, ρ = ρ, ϵ = ϵ), p -> invdecay(p, decay), p -> descent(p, 1))

"""
    ADAM(params, η = 0.001; β1 = 0.9, β2 = 0.999, ϵ = 1e-08, decay = 0)

[ADAM](https://arxiv.org/abs/1412.6980v8) optimiser.
"""
ADAM(ps, η = 0.001; β1 = 0.9, β2 = 0.999, ϵ = 1e-08, decay = 0) =
  optimiser(ps, p -> adam(p; η = η, β1 = β1, β2 = β2, ϵ = ϵ), p -> invdecay(p, decay), p -> descent(p, 1))

"""
    ADAGrad(params; η = 0.01, ϵ = 1e-8, decay = 0)

[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimiser.
Parameters don't need tuning.
"""
ADAGrad(ps; η = 0.01, ϵ = 1e-8, decay = 0) =
  optimiser(ps, p -> adagrad(p; η = η, ϵ = ϵ), p -> invdecay(p, decay), p -> descent(p, 1))

"""
    ADADelta(params; η = 0.01, ρ = 0.95, ϵ = 1e-8, decay = 0)

[ADADelta](http://arxiv.org/abs/1212.5701) optimiser. Parameters don't need
tuning.
"""
ADADelta(ps; η = 0.01, ρ = 0.95, ϵ = 1e-8, decay = 0) =
  optimiser(ps, p -> adadelta(p; ρ = ρ, ϵ = ϵ), p -> invdecay(p, decay), p -> descent(p, 1))

"""
    AMSGrad(params; η = 0.001, β1 = 0.9, β2 = 0.999, ϵ = 1e-08, decay = 0)

[AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) optimiser. Parameters don't need
tuning.
"""
AMSGrad(ps, η = 0.001; β1 = 0.9, β2 = 0.999, ϵ = 1e-08, decay = 0) =
  optimiser(ps, p -> amsgrad(p; η = η, β1 = β1, β2 = β2, ϵ = ϵ), p -> invdecay(p, decay), p -> descent(p, 1))
