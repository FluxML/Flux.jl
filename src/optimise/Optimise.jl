module Optimise

export train!,
	SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,
	InvDecay, ExpDecay, WeightDecay, stop, Optimiser,
	L1_regularization, L2_regularization

include("optimisers.jl")
include("regularization.jl")
include("train.jl")
include("deprecations.jl")

end
