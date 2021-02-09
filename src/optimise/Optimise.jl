module Optimise

using LinearAlgebra
using Optimisers

export train!, skip, stop
       Descent, ADAM, Momentum, Nesterov, RMSProp, 
       ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,RADAM, OADAM, AdaBelief,
       InvDecay, ExpDecay, WeightDecay, stop, skip, ChainOptimiser

include("train.jl")

end
