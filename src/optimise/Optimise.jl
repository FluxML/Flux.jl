module Optimise

using LinearAlgebra

export train!, update!,
	Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,RADAM, 
	InvDecay, ExpDecay, WeightDecay, stop, Optimiser,
	ClipValue, ClipNorm

include("optimisers.jl")
include("train.jl")

end
