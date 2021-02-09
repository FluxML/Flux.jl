module Optimise

using LinearAlgebra
using Optimisers

export train!, 
       ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,RADAM, OADAM, AdaBelief,
       InvDecay, ExpDecay, WeightDecay, stop, skip, Optimiser,
       ClipValue, ClipNorm

include("train.jl")

end
