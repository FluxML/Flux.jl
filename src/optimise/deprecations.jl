using Base: depwarn

function check_decay(opt, decay)
  if decay == 0.
    opt = opt
  else
    if opt isa ADAMW
      opt = Compose(opt, DescentWeightDecay(1, decay))
    else
      opt = Compose(opt, InvDecay(decay))
    end
  end
  opt
end

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
  depwarn("Descent(ps::Param) is deprecated; use Descent(η::Float64) instead", :Descent)

  ps = params
  opt = Descent(η)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function Momentum(params::AbstractArray, η = 0.01; ρ = 0.9, decay = 0.)
  depwarn("Momentum(ps::Param) is deprecated; use Momentum(η::Float64) instead", :Momentum)

  ps = params
  opt = Momentum(η, ρ)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function Nesterov(params::AbstractArray, η = 0.001; ρ = 0.9, decay = 0.)
  depwarn("Nesterov(ps::Param) is deprecated; use Nesterov(η::Float64) instead", :Nesterov)

  ps = params
  opt = Nesterov(η, ρ)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function RMSProp(params::AbstractArray, η = 0.001; ρ = 0.9, decay = 0.)
  depwarn("RMSProp(ps::Param) is deprecated; use RMSProp(η::Float64) instead", :RMSProp)

  ps = params
  opt = RMSProp(η, ρ)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function ADAM(params::AbstractArray, η = 0.001; β1 = 0.9, β2 = 0.999, decay = 0.)
  depwarn("ADAM(ps::Param) is deprecated; use ADAM(η::Float64) instead", :ADAM)

  ps = params
  β = (β1, β2)
  opt = ADAM(η, β)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function ADAGrad(params::AbstractArray, η::Float64 = 0.1; decay = 0.)
  depwarn("ADAGrad(ps::Param) is deprecated; use ADAGrad(η::Float64) instead", :ADAGrad)

  ps = params
  opt = ADAGrad(η)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function ADADelta(params::AbstractArray, ρ::Float64 = 0.9; decay = 0.)
  depwarn("ADADelta(ps::Param) is deprecated; use ADADelta(η::Float64) instead", :ADADelta)

  ps = params
  opt = ADADelta(ρ)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function AdaMax(params::AbstractArray, η = 0.001; β1 = 0.9, β2 = 0.999, decay = 0.)
  depwarn("AdaMax(ps::Param) is deprecated; use AdaMax(η::Float64) instead", :AdaMax)

  ps = params
  β = (β1, β2)
  opt = AdaMax(η, β)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function AMSGrad(params::AbstractArray, η = 0.001; β1 = 0.9, β2 = 0.999, decay = 0.)
  depwarn("AMSGrad(ps::Param) is deprecated; use AMSGrad(η::Float64) instead", :AMSGrad)

  ps = params
  β = (β1, β2)
  opt = AMSGrad(η, β)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function NADAM(params::AbstractArray, η = 0.001; β1 = 0.9, β2 = 0.999, decay = 0.)
  depwarn("NADAM(ps::Param) is deprecated; use NADAM(η::Float64) instead", :NADAM)

  ps = params
  β = (β1, β2)
  opt = NADAM(η, β)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end

function ADAMW(params::AbstractArray, η = 0.001; β1 = 0.9, β2 = 0.999, decay = 0.)
  depwarn("ADAMW(ps::Param) is deprecated; use ADAMW(η::Float64) instead", :ADAMW)

  ps = params
  β = (β1, β2)
  opt = ADAMW(η, β)
  opt = check_decay(opt, decay)
  updaterule(opt, ps)
end