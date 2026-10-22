module Losses

using Statistics
using Zygote
using Zygote: @adjoint
using ChainRulesCore
using ..Flux: BFloat16, _to_bf16
ofeltype(x, y) = convert(float(eltype(x)), y)
epseltype(x) = eps(_upcasttype(float(eltype(x))))
using NNlib: NNlib, logsoftmax, logsumexp, logσ, ctc_loss, ctc_alpha, ∇ctc_loss
using OneHotArrays: OneHotArray, _indices
import Base.Broadcast: broadcasted

export mse, mae, msle,
    label_smoothing,
    crossentropy, logitcrossentropy,
    binarycrossentropy, logitbinarycrossentropy,
    kldivergence,
    huber_loss,
    tversky_loss,
    dice_coeff_loss,
    poisson_loss,
    hinge_loss, squared_hinge_loss,
    binary_focal_loss, focal_loss, siamese_contrastive_loss

include("utils.jl")
include("functions.jl")

end #module
