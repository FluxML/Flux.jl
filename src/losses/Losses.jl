module Losses

using Statistics
using Zygote
using Zygote: @adjoint
using ChainRulesCore
using ..Flux: ofeltype, epseltype, _greek_ascii_depwarn
using CUDA
using Adapt
using MLUtils: ones_like
using NNlib: logsoftmax, logσ, ctc_loss, ctc_alpha, ∇ctc_loss, conv, pad_symmetric
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
    binary_focal_loss, focal_loss, siamese_contrastive_loss,
    ssim, ssim_loss, ssim_loss_fast

include("utils.jl")
include("functions.jl")

end #module
