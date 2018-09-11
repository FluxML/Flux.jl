module Optimise

export train!,
	Descent, ADAM, Momentum, Nesterov, RMSProp

include("optimisers.jl")
include("train.jl")

end