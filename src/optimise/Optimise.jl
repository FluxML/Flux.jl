module Optimise

export train!,
	Descent, ADAM, Momentum, Nesterov, RMSProp, stop, StopException

include("optimisers.jl")
include("train.jl")

end