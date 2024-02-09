module Optimise

using LinearAlgebra

export train!, update!,
	Descent, Adam, Momentum, Nesterov, RMSProp,
	AdaGrad, AdaMax, AdaDelta, AMSGrad, NAdam, AdamW,RAdam, OAdam, AdaBelief,
	InvDecay, ExpDecay, WeightDecay, Optimiser,
	ClipValue, ClipNorm, SignDecay

include("optimisers.jl")
include("train.jl")

end
