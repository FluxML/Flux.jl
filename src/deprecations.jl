
# v0.13 deprecations

# Channel notation: Changed to match Conv, but very softly deprecated!
# Perhaps change to @deprecate for v0.15, but there is no plan to remove these.
Dense(in::Integer, out::Integer, σ = identity; kw...) =
  Dense(in => out, σ; kw...)
Bilinear(in1::Integer, in2::Integer, out::Integer, σ = identity; kw...) =
  Bilinear((in1, in2) => out, σ; kw...)
Embedding(in::Integer, out::Integer; kw...) = Embedding(in => out; kw...)

RNNCell(in::Integer, out::Integer, σ = tanh; kw...) = RNNCell(in => out, σ; kw...)
LSTMCell(in::Integer, out::Integer; kw...) = LSTMCell(in => out; kw...)

GRUCell(in::Integer, out::Integer; kw...) = GRUCell(in => out; kw...)
GRUv3Cell(in::Integer, out::Integer; kw...) = GRUv3Cell(in => out; kw...)


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

train!(loss, model, data, opt::Optimise.AbstractOptimiser; cb=nothing) =
  train!(loss, model, data, _old_to_new(opt); cb)

# Next, to use the new `setup` with the still-exported old-style `Adam` etc:
import .Train: setup
setup(rule::Optimise.AbstractOptimiser, model) = setup(_old_to_new(rule), model)
# ... and allow accidental use of `Optimisers.setup` to do the same:
Optimisers.setup(rule::Optimise.AbstractOptimiser, model) = setup(_old_to_new(rule), model)

for T in [:Descent, :Adam, :Momentum, :Nesterov,
   	      :AdaGrad, :AdaMax, :AdaDelta, :AMSGrad, :NAdam, :RAdam, :OAdam, :AdaBelief,
   	      # :InvDecay, :ExpDecay, 
          :SignDecay,
          ]
  @eval function _old_to_new(rule::$T)
    args = map(f -> getfield(rule, f), fieldnames(Optimisers.$T))
    Optimisers.$T(args...)
  end
end
_old_to_new(rule::Optimiser) = Optimisers.OptimiserChain(map(_old_to_new, rule.os)...)
const OptimiserChain = Optimise.Optimiser  # lets you use new name with implicit params too.
_old_to_new(rule::WeightDecay) = Optimisers.WeightDecay(rule.wd)  # called lambda now
_old_to_new(rule::ClipNorm) = Optimisers.ClipNorm(rule.thresh)  # called omega, and there are more fields 
_old_to_new(rule::ClipValue) = Optimisers.ClipGrad(rule.thresh)  # called delta now, and struct name differs
const ClipGrad = Optimise.ClipValue
_old_to_new(rule::RMSProp) = Optimisers.RMSProp(rule.eta, rule.rho, rule.epsilon)  # RMSProp has no field centred

_old_to_new(rule) = error("Flux.setup does not know how to translate this old-style implicit rule to a new-style Optimisers.jl explicit rule")

# This allows you to mix and match, like Flux.setup(OptimiserChain(Optimisers.SignDecay(), Flux.Descent()), [1,2,3.])
Optimisers.OptimiserChain(rules::Union{Optimisers.AbstractRule, Optimise.AbstractOptimiser}...) =
  Optimisers.OptimiserChain(map(_old_to_new, rules))
_old_to_new(rule::Optimisers.AbstractRule) = rule

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
@deprecate default_rng_value() Random.default_rng()


# Issue 2476, after ConvTranspose got a new field in 2462. Minimal fix to allow loading?
function loadmodel!(dst::ConvTranspose, src::NamedTuple{(:σ, :weight, :bias, :stride, :pad, :dilation, :groups)}; kw...)
  new_src = (; src.σ, src.weight, src.bias, src.stride, src.pad, dst.outpad, src.dilation, src.groups)
  loadmodel!(dst, new_src; kw...)
end

function get_device(; verbose::Bool=false)
  Base.depwarn("get_device() is deprecated. Use `gpu_device()` instead.", :get_device)
  return MLDataDevices.gpu_device()
end

function get_device(backend::String, idx::Int = 0)
  Base.depwarn("get_device(backend::String, idx::Int) is deprecated. Use `gpu_device(idx+1)` instead.", :get_device)
  if backend == "AMD"
      @warn "\"AMD\" backend is deprecated. Please use \"AMDGPU\" instead." maxlog=1
      backend = "AMDGPU"
  end
  if backend == "CPU"
      return cpu_device()
  else
      return gpu_device(idx+1, force=true)
  end
end

function supported_devices()
  Base.depwarn("`supported_devices()` is deprecated. Use `supported_gpu_backends()` instead.", :supported_devices)
  return MLDataDevices.supported_gpu_backends()
end

# This was previosly documented. 
# As of v0.14.23 we silently deprecate it. 
# Later we will deprecate it loudly and then remove it.
const GPU_BACKEND = @load_preference("gpu_backend", "CUDA") 


# help out with https://github.com/chengchingwen/Transformers.jl/issues/201
const FluxCPUAdaptor = CPUDevice
const FluxCUDAAdaptor = CUDADevice
const FluxAMDGPUAdaptor = AMDGPUDevice
const FluxMetalAdaptor = MetalDevice

# v0.15 deprecations

# Enable these when 0.15 is released, and delete const ClipGrad = Optimise.ClipValue etc: 
# Base.@deprecate_binding Optimiser OptimiserChain
# Base.@deprecate_binding ClipValue ClipGrad

# train!(loss::Function, ps::Zygote.Params, data, opt) = throw(ArgumentError(
#   """On Flux 0.15, `train!` no longer accepts implicit `Zygote.Params`.
#   Instead of `train!(loss_xy, Flux.params(model), data, Adam())`
#   it now needs `opt = Flux.setup(Adam(), model); train!(loss_mxy, model, data, opt)`
#   where `loss_mxy` accepts the model as its first argument.
#   """
# ))
