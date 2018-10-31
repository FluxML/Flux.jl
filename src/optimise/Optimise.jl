module Optimise

export train!,
	SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,
	InvDecay, ExpDecay, WeightDecay, stop, Optimiser

include("optimisers.jl")
include("train.jl")
include("deprecations.jl")

end
