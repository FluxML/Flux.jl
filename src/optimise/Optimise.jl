module Optimise

using LinearAlgebra
using ArrayInterface: ArrayInterface

export train!,
    update!,
    Descent,
    ADAM,
    Momentum,
    Nesterov,
    RMSProp,
    ADAGrad,
    AdaMax,
    ADADelta,
    AMSGrad,
    NADAM,
    ADAMW,
    RADAM,
    OADAM,
    AdaBelief,
    InvDecay,
    ExpDecay,
    WeightDecay,
    stop,
    skip,
    Optimiser,
    ClipValue,
    ClipNorm

include("optimisers.jl")
include("train.jl")

end
