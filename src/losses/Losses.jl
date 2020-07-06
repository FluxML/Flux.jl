module Losses

using Statistics
using Zygote
using Zygote: @adjoint
using ..Flux: ofeltype, epseltype
using CUDA
using NNlib: logsoftmax, logσ
import Base.Broadcast: broadcasted

export mse, mae, msle,
    crossentropy, logitcrossentropy,
    # binarycrossentropy, logitbinarycrossentropy # export only after end deprecation
    kldivergence,
    huber_loss,
    tversky_loss,
    dice_coeff_loss,
    poisson_loss,
    hinge_loss, squared_hinge_loss

include("utils.jl")
include("functions.jl")

end #module