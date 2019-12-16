module Optimise

using LinearAlgebra
import ArrayInterface

export train!, update!,
	Descent, ADAM, Momentum, Nesterov, RMSProp, Lookahead,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,RADAM, OADAM, AdaBelief,
	InvDecay, ExpDecay, WeightDecay, stop, skip, Optimiser,
	ClipValue, ClipNorm

include("optimisers.jl")
include("lookahead.jl")
include("train.jl")

end
