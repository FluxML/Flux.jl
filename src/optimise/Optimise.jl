module Optimise

export train!,
	Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,
	InvDecay, ExpDecay, stop, Compose

include("optimisers.jl")
include("train.jl")
include("deprecations.jl")

end
