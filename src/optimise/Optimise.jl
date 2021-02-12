module Optimise

using LinearAlgebra

export train!, update!,
	Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,RADAM, OADAM, AdaBelief,
	InvDecay, ExpDecay, WeightDecay, stop, skip, Optimiser,
	ClipValue, ClipNorm

include("optimisers.jl")

module Schedule
	using ..Optimise
	using ParameterSchedulers
	import ParameterSchedulers: AbstractSchedule

	include("schedulers.jl")
end

include("train.jl")

end
