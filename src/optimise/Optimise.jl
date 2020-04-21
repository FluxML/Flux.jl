module Optimise

using LinearAlgebra

export train!, update!, stop, Optimiser,
		Descent, ADAM, Momentum, Nesterov, RMSProp,
		ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM,
		InvDecay, ExpDecay, WeightDecay, ClipValue, ClipNorm

include("optimisers.jl")
include("train.jl")

end
