module Optimise

using LinearAlgebra

export train!, update!,
	SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, RADAM, 
	InvDecay, ExpDecay, WeightDecay, ClipValue, ClipNorm,
	stop, Optimiser

include("optimisers.jl")
include("train.jl")

end
