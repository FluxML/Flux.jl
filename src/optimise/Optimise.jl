module Optimise

using LinearAlgebra
using Optimisers
using Optimisers: apply

export train!,
       Descent, ADAM, Momentum, Nesterov, RMSProp, 
       ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,RADAM, OADAM, AdaBelief,
       WeightDecay, stop, skip, ChainOptimiser

include("train.jl")

end
