module Optimise

using LinearAlgebra
using Optimisers
using Optimisers: Descent, ADAM, Momentum, Nesterov, RMSProp

export train!, update!,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,RADAM, OADAM, AdaBelief,
	InvDecay, ExpDecay, WeightDecay, stop, skip, Optimiser,
	ClipValue, ClipNorm

include("optimisers.jl")
include("train.jl")

end
