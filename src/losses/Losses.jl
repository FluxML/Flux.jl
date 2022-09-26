"""
    Flux.Losses
    
This sub-module contains many loss functions, all of which accept two arguments,
with the model output as the fist argument: `loss(model(x), y)`.
It also contains a few related utilities, such as `label_smoothing`.
The complete list of exports is:

    label_smoothing,
    mse, mae, msle,
    crossentropy,
    logitcrossentropy,
    binarycrossentropy,
    logitbinarycrossentropy,
    kldivergence,
    huber_loss,
    tversky_loss,
    dice_coeff_loss,
    poisson_loss,
    hinge_loss,
    squared_hinge_loss,
    binary_focal_loss,
    focal_loss,
    siamese_contrastive_loss
"""
module Losses

using Statistics
using Zygote
using Zygote: @adjoint
using ChainRulesCore
using ..Flux: ofeltype, epseltype
using CUDA
using NNlib: logsoftmax, logσ, ctc_loss, ctc_alpha, ∇ctc_loss
import Base.Broadcast: broadcasted

export label_smoothing,
    mse, mae, msle,
    crossentropy, logitcrossentropy,
    binarycrossentropy, logitbinarycrossentropy,
    kldivergence,
    huber_loss,
    tversky_loss,
    dice_coeff_loss,
    poisson_loss,
    hinge_loss, squared_hinge_loss,
    binary_focal_loss, focal_loss,
    siamese_contrastive_loss

include("utils.jl")
include("functions.jl")

for loss in Symbol.([
  mse, mae, msle,
  crossentropy, logitcrossentropy,
  binarycrossentropy, logitbinarycrossentropy,
  kldivergence,
  huber_loss,
  tversky_loss,
  dice_coeff_loss,
  poisson_loss,
  hinge_loss, squared_hinge_loss,
  binary_focal_loss, focal_loss,
  siamese_contrastive_loss,
  ])
  @eval begin
    """
        $($loss)(model, x, y)
  
    This method calculates `ŷ = model(x)`. Accepts the same keyword arguments.
    """
    $loss(f, x::AbstractArray, y::AbstractArray; kw...) = $loss(f(x), y; kw...)
  end
end

end #module
