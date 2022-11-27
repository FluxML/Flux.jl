# v0.12 deprecations

function ones(dims...)
  Base.depwarn("Flux.ones(size...) is deprecated, please use Flux.ones32(size...) or Base.ones(Float32, size...)", :ones, force=true)
  Base.ones(Float32, dims...)
end
ones(T::Type, dims...) = Base.ones(T, dims...)

function zeros(dims...)
  Base.depwarn("Flux.zeros(size...) is deprecated, please use Flux.zeros32(size...) or Base.zeros(Float32, size...)", :zeros, force=true)
  Base.zeros(Float32, dims...)
end
zeros(T::Type, dims...) = Base.zeros(T, dims...)

ones32(::Type, dims...) = throw(ArgumentError("Flux.ones32 is always Float32, use Base.ones to specify the element type"))
zeros32(::Type, dims...) = throw(ArgumentError("Flux.zeros32 is always Float32, use Base.zeros to specify the element type"))

# v0.13 deprecations

function Broadcast.broadcasted(f::Recur, args...)
  # This had an explicit @adjoint rule, calling Zygote.∇map(__context__, f, args...), until v0.12
  Base.depwarn("""Broadcasting is not safe to use with RNNs, as it does not guarantee an iteration order.
    Re-writing this as a comprehension would be better.""", :broadcasted)
  map(f, args...)  # map isn't really safe either, but 
end

@deprecate frequencies(xs) group_counts(xs)

struct Zeros
  function Zeros()
    Base.depwarn("Flux.Zeros is no more, has ceased to be, is bereft of life, is an ex-boondoggle... please use bias=false instead", :Zeros)
    false
  end
end
Zeros(args...) = Zeros()  # was used both Dense(10, 2, initb = Zeros) and Dense(rand(2,10), Zeros())

function Optimise.update!(x::AbstractArray, x̄)
  Base.depwarn("`Flux.Optimise.update!(x, x̄)` was not used internally and has been removed. Please write `x .-= x̄` instead.", :update!)
  x .-= x̄
end

function Diagonal(size::Integer...; kw...)
  Base.depwarn("Flux.Diagonal is now Flux.Scale, and also allows an activation function.", :Diagonal)
  Scale(size...; kw...)
end
function Diagonal(size::Tuple; kw...)
  Base.depwarn("Flux.Diagonal is now Flux.Scale, and also allows an activation function.", :Diagonal)
  Scale(size...; kw...)
end

# Deprecate this eventually once saving models w/o structure is no more
function loadparams!(m, xs)
  Base.depwarn("loadparams! will be deprecated eventually. Use loadmodel! instead.", :loadparams!)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end

# Channel notation: Changed to match Conv, but very softly deprecated!
# Perhaps change to @deprecate for v0.14, but there is no plan to remove these.
Dense(in::Integer, out::Integer, σ = identity; kw...) =
  Dense(in => out, σ; kw...)
Bilinear(in1::Integer, in2::Integer, out::Integer, σ = identity; kw...) =
  Bilinear((in1, in2) => out, σ; kw...)
Embedding(in::Integer, out::Integer; kw...) = Embedding(in => out; kw...)

RNNCell(in::Integer, out::Integer, σ = tanh; kw...) = RNNCell(in => out, σ; kw...)
LSTMCell(in::Integer, out::Integer; kw...) = LSTMCell(in => out; kw...)

GRUCell(in::Integer, out::Integer; kw...) = GRUCell(in => out; kw...)
GRUv3Cell(in::Integer, out::Integer; kw...) = GRUv3Cell(in => out; kw...)

# Optimisers with old naming convention
Base.@deprecate_binding ADAM Adam
Base.@deprecate_binding NADAM NAdam
Base.@deprecate_binding ADAMW AdamW
Base.@deprecate_binding RADAM RAdam
Base.@deprecate_binding OADAM OAdam
Base.@deprecate_binding ADAGrad AdaGrad
Base.@deprecate_binding ADADelta AdaDelta

# Remove sub-module Data, while making sure Flux.Data.DataLoader keeps working
Base.@deprecate_binding Data Flux false "Sub-module Flux.Data has been removed. The only thing it contained may be accessed as Flux.DataLoader"

@deprecate rng_from_array() default_rng_value()

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
_old_to_new(rule::ClipNorm) = Optimisers.ClipNorm(rule.thesh)  # called omega, and there are more fields 
_old_to_new(rule::ClipValue) = Optimisers.ClipGrad(rule.thesh)  # called delta now, and struct name differs
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

# v0.14 deprecations

# Enable these when 0.14 is released, and delete const ClipGrad = Optimise.ClipValue etc: 
# Base.@deprecate_binding Optimiser OptimiserChain
# Base.@deprecate_binding ClipValue ClipGrad

# train!(loss::Function, ps::Zygote.Params, data, opt) = throw(ArgumentError(
#   """On Flux 0.14, `train!` no longer accepts implicit `Zygote.Params`.
#   Instead of `train!(loss_xy, Flux.params(model), data, Adam())`
#   it now needs `opt = Flux.setup(Adam(), model); train!(loss_mxy, model, data, opt)`
#   where `loss_mxy` accepts the model as its first argument.
#   """
# ))
