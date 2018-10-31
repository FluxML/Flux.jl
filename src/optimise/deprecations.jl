using Base: depwarn

check_decay(opt, decay) = decay == 0 ? opt : Optimiser(opt, InvDecay(decay))

# legacy update rule
function updaterule(opt, ps)
  () -> begin
    for p in ps
      delta = update!(opt, p.data, p.grad)
      p.data .-= delta
    end
  end
end

function Descent(params::AbstractArray, η = 0.1; decay = 0.)
  depwarn("Descent(params) is deprecated; use Descent(η::Float64) instead", :Descent)

  ps = params
  opt = Descent(η)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function Momentum(params::AbstractArray, η = 0.01; ρ = 0.9, decay = 0.)
  depwarn("Momentum(params) is deprecated; use Momentum(η::Float64) instead", :Momentum)

  ps = params
  opt = Momentum(η, ρ)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function Nesterov(params::AbstractArray, η = 0.001; ρ = 0.9, decay = 0.)
  depwarn("Nesterov(params) is deprecated; use Nesterov(η::Float64) instead", :Nesterov)

  ps = params
  opt = Nesterov(η, ρ)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function RMSProp(params::AbstractArray, η = 0.001; ρ = 0.9, decay = 0.)
  depwarn("RMSProp(params) is deprecated; use RMSProp(η::Float64) instead", :RMSProp)

  ps = params
  opt = RMSProp(η, ρ)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function ADAM(params::AbstractArray, η = 0.001; β1 = 0.9, β2 = 0.999, decay = 0.)
  depwarn("ADAM(params) is deprecated; use ADAM(η::Float64) instead", :ADAM)

  ps = params
  β = (β1, β2)
  opt = ADAM(η, β)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function ADAGrad(params::AbstractArray, η::Float64 = 0.1; decay = 0.)
  depwarn("ADAGrad(params) is deprecated; use ADAGrad(η::Float64) instead", :ADAGrad)

  ps = params
  opt = ADAGrad(η)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function ADADelta(params::AbstractArray, ρ::Float64 = 0.9; decay = 0.)
  depwarn("ADADelta(params) is deprecated; use ADADelta(η::Float64) instead", :ADADelta)

  ps = params
  opt = ADADelta(ρ)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function AdaMax(params::AbstractArray, η = 0.001; β1 = 0.9, β2 = 0.999, decay = 0.)
  depwarn("AdaMax(params) is deprecated; use AdaMax(η::Float64) instead", :AdaMax)

  ps = params
  β = (β1, β2)
  opt = AdaMax(η, β)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function AMSGrad(params::AbstractArray, η = 0.001; β1 = 0.9, β2 = 0.999, decay = 0.)
  depwarn("AMSGrad(params) is deprecated; use AMSGrad(η::Float64) instead", :AMSGrad)

  ps = params
  β = (β1, β2)
  opt = AMSGrad(η, β)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function NADAM(params::AbstractArray, η = 0.001; β1 = 0.9, β2 = 0.999, decay = 0.)
  depwarn("NADAM(params) is deprecated; use NADAM(η::Float64) instead", :NADAM)

  ps = params
  β = (β1, β2)
  opt = NADAM(η, β)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function ADAMW(params::AbstractArray, η = 0.001; β1 = 0.9, β2 = 0.999, decay = 0.)
  depwarn("ADAMW(params) is deprecated; use ADAMW(η::Float64) instead", :ADAMW)

  ps = params
  β = (β1, β2)
  opt = ADAMW(η, β)
  opt = check_decay(opt, decay)
  decay != 0 && (opt = Optimiser(opt, WeightDecay(decay)))
  updaterule(opt, ps)
end

# Old training loop

struct OldOptimiser
  func
end

update!(opt::OldOptimiser, ps) = opt.func()

# Train function
function train!(loss, data, opt; cb = () -> ())
  depwarn("train!(loss, data, opt) is deprecated; use train!(loss, params, data, opt) instead", :train!)
  train!(loss, (), data, OldOptimiser(opt); cb = cb)
end
