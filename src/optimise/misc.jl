export SGD, AdaGrad, RMSProp, AdaDelta, Adam

struct Optimizer
  steps
end

function (o::Optimizer)(ps::Vector{Param})
  states = map(ps) do p
    p, map(x->x(p), o.steps)
  end

  () -> for (p, steps) in states
    foreach(f->f(p), steps)
    @. p.x -= p.Δx
  end
end

function Momentum(η)
  function (p)
    momentum = zeros(p.x)

    function (p)
      @. momentum = η * momentum + p.Δx
      @. p.Δx = momentum
    end
  end
end

function NesterovMomentum(η)
  function (p)
    momentum = zeros(p.x)

    function (p)
      @. momentum = η * momentum + p.Δx
      @. p.Δx = η * momentum + p.Δx
    end
  end
end

function WeightDecayConst(γ)
  function (p)
    function (p)
      # avoid bouncing around 0
      x = p.x - p.Δx
      @. p.Δx += (abs(x) <= γ) * x + (abs(x) > γ) * γ * sign(x)
    end
  end
end

function WeightDecayRatio(γ)
  function (p)
    function (p)
      @. p.Δx += γ * p.x
    end
  end
end

function GradDecayFix(lr)
  function (p)
    function (p)
      @. p.Δx *= lr
    end
  end
end

function GradDecayExp(γ)
  function (p)
    n_iter = 0

    function (p)
      p.Δx .*= γ ^ n_iter
      n_iter += 1
    end
  end
end

function GradDecayInv(γ)
  function (p)
    n_iter = 0

    function (p)
      p.Δx .*= 1 / (1 + γ * n_iter)
      n_iter += 1
    end
  end
end

function GradClipConst(threshold)
  function (p)
    function (p)
      p.Δx .= max.(min.(p.Δx, threshold), -threshold)
    end
  end
end

function Accumulate(window)
  function (p)
    index = 0
    acc = zeros(p.x)

    function (p)
      acc .+= p.Δx

      if index >= window
        p.Δx .= acc
        acc .= 0
        index = 0
      else
        p.Δx .= 0
        index += 1
      end
    end
  end
end

function _AdaGrad(ϵ)
  function (p)
    acc = zeros(p.x) .+ ϵ

    function (p)
      @. acc += p.Δx ^ 2
      @. p.Δx /= √acc
    end
  end
end

function _RMSProp(ρ, ϵ)
  function (p)
    acc  = zeros(p.x) .+ ϵ

    function (p)
      @. acc = ρ * acc + (1 - ρ) * p.Δx ^ 2
      @. p.Δx /= √acc
    end
  end
end

function _AdaDelta(ρ, ϵ)
  function (p)
    acc = zeros(p.x) .+ ϵ
    Δacc = zeros(p.x) .+ ϵ

    function (p)
      @. acc = ρ * acc + (1 - ρ) * p.Δx ^ 2
      @. p.Δx *= √Δacc / √acc
      @. Δacc = ρ * Δacc + (1 - ρ) * p.Δx ^ 2
    end
  end
end

function _Adam(β1, β2, ϵ)
  function (p)
    mt = zeros(p.x)
    vt = zeros(p.x) .+ ϵ
    β1p = β1
    β2p = β2

    function (p)
      @. mt = β1 * mt + (1 - β1) * p.Δx
      @. vt = β2 * vt + (1 - β2) * p.Δx ^ 2

      @. p.Δx = √(1 - β2p) / √(1 - β1p) * mt / √vt

      β1p *= β1
      β2p *= β2
    end
  end
end

macro restrict_range(var::Symbol, range::String)
  left, right = split(range, ", ")
  lo = left[1] == '[' ? :>= : :>
  lt = left[2:end]
  ro = right[end] == ']' ? :<= : :<
  rt = right[1:end-1]

  error_msg = "$var ∈ $range must be hold"
  var = esc(var)

  quote
    $( lt != "-∞" && :( $lo($var, $(parse(Float64, lt))) || throw(ArgumentError($error_msg)) ) )
    $( rt != "∞"  && :( $ro($var, $(parse(Float64, rt))) || throw(ArgumentError($error_msg)) ) )
  end
end

function SGD(; lr::Real=.1,
               momentum::Real=0,
               decay::Real=0,
               nesterov::Bool=false)

  @restrict_range lr       "[0, ∞)"
  @restrict_range momentum "[0, 1]"
  @restrict_range decay    "[0, ∞)"

  steps = []

  if momentum != 0
    nesterov ? push!(steps, NesterovMomentum(momentum)) :
               push!(steps, Momentum(momentum))
  end

  decay != 0 && push!(steps, GradDecayInv(decay))

  lr != 1 && push!(steps, GradDecayFix(lr))

  Optimizer(steps)
end

function AdaGrad(; lr::Real=.001,
                   epsilon::Real=1e-6,
                   decay::Real=0.)

  @restrict_range lr      "[0, ∞)"
  @restrict_range epsilon "(0, ∞)"
  @restrict_range decay   "[0, ∞)"

  steps = Any[_AdaGrad(epsilon)]

  decay != 0 && push!(steps, GradDecayInv(decay))

  lr != 1 && push!(steps, GradDecayFix(lr))

  Optimizer(steps)
end

function RMSProp(; lr::Real=.001,
                   rho::Real=.9,
                   epsilon::Real=1e-6,
                   decay::Real=0.)

  @restrict_range lr      "[0, ∞)"
  @restrict_range rho     "[0, 1]"
  @restrict_range epsilon "(0, ∞)"
  @restrict_range decay   "[0, ∞)"

  steps = Any[_RMSProp(rho, epsilon)]

  decay != 0 && push!(steps, GradDecayInv(decay))

  lr != 1 && push!(steps, GradDecayFix(lr))

  Optimizer(steps)
end

function AdaDelta(; lr::Real=1.,
                    rho::Real=.9,
                    epsilon::Real=1e-6,
                    decay::Real=0.)

  @restrict_range lr      "[0, ∞)"
  @restrict_range rho     "[0, 1]"
  @restrict_range epsilon "(0, ∞)"
  @restrict_range decay   "[0, ∞)"

  steps = Any[_AdaDelta(rho, epsilon)]

  decay != 0 && push!(steps, GradDecayInv(decay))

  lr != 1 && push!(steps, GradDecayFix(lr))

  Optimizer(steps)
end

function Adam(; lr::Real=.1,
                beta1::Real=.9,
                beta2::Real=.999,
                epsilon::Real=1e-6,
                decay::Real=0.)

  @restrict_range lr      "[0, ∞)"
  @restrict_range beta1   "[0, 1]"
  @restrict_range beta2   "[0, 1]"
  @restrict_range epsilon "(0, ∞)"
  @restrict_range decay   "[0, ∞)"

  steps = Any[_Adam(beta1, beta2, epsilon)]

  decay != 0 && push!(steps, GradDecayInv(decay))

  lr != 1 && push!(steps, GradDecayFix(lr))

  Optimizer(steps)
end
