module Optimise

using Flux
using MacroTools: @forward
import Zygote
import Zygote: Params, gradient
import Tracker
using AbstractDifferentiation
import Optimisers
import Optimisers: update, update!
using LinearAlgebra
import ArrayInterface
using ProgressLogging: @progress, @withprogress, @logprogress

export train!, update!,
	Descent, Adam, Momentum, Nesterov, RMSProp,
	AdaGrad, AdaMax, AdaDelta, AMSGrad, NAdam, AdamW,RAdam, OAdam, AdaBelief,
	InvDecay, ExpDecay, WeightDecay, stop, skip, Optimiser,
	ClipValue, ClipNorm

include("optimisers.jl")
include("gradients.jl")
include("train.jl")

end
