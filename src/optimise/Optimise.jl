module Optimise

export train!,
	SGD, Descent, ADAM, AdaMax, Momentum, Nesterov, RMSProp, ADAGrad, ADADelta, AMSGrad

include("optimisers.jl")
include("train.jl")

end