
#=
  # Valid method in Optimise, old implicit style, is:
  train!(loss, ps::Params, data, opt::AbstractOptimiser; cb = () -> ())

  # Valid methods in Train, new explict style, are:
  train!(loss, model, data, opt)  # preferred
  train!(loss, model, data, opt::Optimisers.AbstractRule)  # if you forget setup

  # Provide friendly errors for what happens if you mix these up:
=#
import .Optimise: train!

train!(loss, ps::Params, data, opt; cb=nothing) = error(
  """can't mix implict Params with explict state!
  To use `Flux.params(m)` in `train!`, the 4th argument must be from the old `Flux.Optimise` sub-module.
  But better to use the new explicit style, in which `m` itself is the 2nd argument.
  """)

train!(loss, ps::Params, data, opt::Optimisers.AbstractRule; cb=nothing) = error(
  """can't mix implict Params with explict rule from Optimisers.jl
  To use `Flux.params(m)` in `train!`, the 4th argument must be from the old `Flux.Optimise` sub-module.
  But better to use the new explicit style, in which `m` itself is the 2nd argument.
  """)

train!(loss, model, data, opt::Optimise.AbstractOptimiser; cb=nothing) = train!(loss, model, data, _old_to_new(opt); cb)

# Next, to use the new `setup` with the still-exported old-style `Adam` etc:
import .Train: setup
setup(rule::Optimise.AbstractOptimiser, model) = setup(_old_to_new(rule), model)
# ... and allow accidental use of `Optimisers.setup` to do the same:
Optimisers.setup(rule::Optimise.AbstractOptimiser, model) = setup(_old_to_new(rule), model)

for T in [:Descent, :Adam, :Momentum, :Nesterov,
   	      :AdaGrad, :AdaMax, :AdaDelta, :AMSGrad, :NAdam, :RAdam, :OAdam, :AdaBelief,
   	      # :InvDecay, :ExpDecay, 
          ]
  @eval function _old_to_new(rule::$T)
    args = map(f -> getfield(rule, f), fieldnames(Optimisers.$T))
    Optimisers.$T(args...)
  end
end
_old_to_new(rule::Optimiser) = Optimisers.OptimiserChain(map(_old_to_new, rule.os)...)
const OptimiserChain = Optimise.Optimiser  # lets you use new name with implicit params too.
_old_to_new(rule::WeightDecay) = Optimisers.WeightDecay(rule.wd)  # called gamma now
_old_to_new(rule::ClipNorm) = Optimisers.ClipNorm(rule.thresh)  # called omega, and there are more fields 
_old_to_new(rule::ClipValue) = Optimisers.ClipGrad(rule.thresh)  # called delta now, and struct name differs
const ClipGrad = Optimise.ClipValue
_old_to_new(rule::RMSProp) = Optimisers.RMSProp(rule.eta, rule.rho, rule.epsilon)  # RMSProp has no field centred

_old_to_new(rule) = error("Flux.setup does not know how to translate this old-style implicit rule to a new-style Optimisers.jl explicit rule")

# Since `update!` should be called in a loop, it makes less sense to call `setup` for you if you forgot.
# But let's make sure that such uses give a helpful error:
import .Optimise: update!

function update!(opt::Optimise.AbstractOptimiser, model, grad)
  # This error method requires narrowing the main worker method of Flux.Optimise
  # to accept only arrays. Remove if this causes problems!
  # update!(opt::Flux.Optimise.AbstractOptimiser, x::AbstractArray, x̄)
  error("""Invalid input to `update!`.
    * For the implicit style, this needs `update(::AbstractOptimiser, ::Params, ::Grads)`
    * For the explicit style, `update(state, model, grad)` needs `state = Flux.setup(opt, model)`.
    """)
end

# An easy error to make is to pass result of explicit gradient(...), not gradient(...)[1]
# Can't catch every case, but can catch many simple Flux models:

function update!(opt, model::Chain, grads::Tuple)
  # Zygote will make a NamedTuple{(:layers,)} for the gradient of Chain, Diffractor a Tangent
  @warn """explicit `update!(opt, model, grad)` wants the gradient for the model alone,
    not the whole tuple from `gradient(m -> loss(m, x, y), model)`. You probably want `grads[1]`."""
  update!(opt, model, grads[1])
end

function update!(opt::Optimise.AbstractOptimiser, model::Chain, grads::Tuple)  # ambiguity
  update!(opt, model, grads[1])  # calls error case "Invalid input" just above
end

# One more easy error to catch is using explicit gradient with `params(m)`:

function update!(opt::Optimise.AbstractOptimiser, ::Params, grads::Union{Tuple, NamedTuple})
  error("""can't mix implicit Params with explicit gradients!
    * For the implicit style, this needs `update(::AbstractOptimiser, ::Params, ::Grads)` with implicit gradient.
    * For the explicit style, `update(state, model, grad)` needs the model itself, and `state = Flux.setup(opt, model)`.
    """)
end
