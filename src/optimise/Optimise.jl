module Optimise

export train!, update!,
	SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,RADAM, 
	InvDecay, ExpDecay, WeightDecay, stop, Optimiser

include("optimisers.jl")
include("train.jl")

end
