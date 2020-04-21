module Optimise

using Requires

export train!, update!,
	Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,RADAM, 
	InvDecay, ExpDecay, WeightDecay, stop, Optimiser

include("optimisers.jl")
include("train.jl")

function __init__()
	@require MPI="da04e1cc-30fd-572f-bb4f-1f8673147195" include("mpi.jl")
end

end
