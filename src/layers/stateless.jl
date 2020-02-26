using CuArrays
using NNlib: logsoftmax, logσ

# Cost functions
"""
    mae(ŷ, y)
L1 loss function. Computes the mean of absolute error between prediction and true values
"""
mae(ŷ, y) = sum(abs.(ŷ .- y)) * 1 // length(y)


"""
    mse(ŷ, y)
L2 loss function. Computes the mean of the squared errors between prediction and true values
"""
mse(ŷ, y) = sum((ŷ .- y).^2) * 1 // length(y)


"""
    mean_squared_logarithmic_error(ŷ, y;ϵ1=eps.(Float64.(ŷ)),ϵ2=eps.(Float64.(y)))

L2 loss function. Returns the mean of the squared logarithmic errors of prediction ŷ, and true values y. The ϵ1 and ϵ2 terms provide numerical stability.
(Computes mean of squared(log(predicted values)-log(true value)). This error penalizes an under-predicted estimate greater than an over-predicted estimate.

  ```julia
  julia> y_=[14726,327378,74734]
  3-element Array{Int64,1}:
    14726
  327378
    74734

  julia> y = [12466.1,16353.95,16367.98]
  3-element Array{Float64,1}:
  12466.1 
  16353.95
  16367.98

  julia> mean_squared_logarithmic_error(y,y_)
  3.771271382334686
  ```
Alias:
  msle(ŷ,y;ϵ1=eps.(Float64.(ŷ)),ϵ2=eps.(Float64.(y)))

"""
mean_squared_logarithmic_error(ŷ, y;ϵ1=eps.(ŷ),ϵ2=eps.(eltype(ŷ).(y))) = sum((log.(ŷ+ϵ1).-log.(y+ϵ2)).^2) * 1 // length(y)
#Alias
msle(ŷ, y;ϵ1=eps.(ŷ),ϵ2=eps.(eltype(ŷ).(y))) = sum((log.(ŷ+ϵ1).-log.(y+ϵ2)).^2) * 1 // length(y)



"""
    huber_loss(ŷ, y,delta=1.0)

Computes the mean of the Huber loss between prediction ŷ and true values y. By default, delta is set to 1.0.
[Huber Loss](https://en.wikipedia.org/wiki/Huber_loss).
  
  ```julia
  julia> y = [1.2636,1.25,1.73]
  3-element Array{Float64,1}:
  1.2636
  1.25  
  1.73  

  julia> y_= [-1.376,0,3.37]
  3-element Array{Float64,1}:
  -1.376
   0.0  
   3.37 

  julia> huber_loss(y,y_)
  0.7131999999999998
  ```

"""
function huber_loss(ŷ, y,delta=1.0)
  abs_error = abs.(ŷ.-y)
  dtype= eltype(ŷ)
  delta = dtype(delta)
  hub_loss = dtype(0)
  for i in 1:length(y)
    if (abs_error[i]<=delta)
      hub_loss+=abs_error[i]^2*dtype(0.5)
    else
      hub_loss+=delta*(abs_error[i]- dtype(0.5*delta))
    end
  
  return hub_loss*1//length(y)
  end
end


function _crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::Nothing)
  return -sum(y .* log.(ŷ)) * 1 // size(y, 2)
end

function _crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::Number)
  return -sum(y .* log.(ŷ)) .* weight * 1 // size(y, 2)
end

function _crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::AbstractVector)
  return -sum(y .* log.(ŷ) .* weight) * 1 // size(y, 2)
end

"""
  crossentropy(ŷ, y, weight)

Computes crossentropy loss over the prediction ŷ and true labels y(expected `onehot` encoded). 'weight' parameter allows to set the class weights while calculating loss.
It can be a number or a vector of class weights. By default, weight is set to nothing.

  ```julia
  julia> ŷ = [0.33 .11 .98;0.11 0.34 0.11]
  2×3 Array{Float64,2}:
  0.33  0.11  0.98
  0.11  0.34  0.11

  julia> y = [1 0 0;0 1 0]
  2×3 Array{Int64,2}:
  1  0  0
  0  1  0

  julia> crossentropy(ŷ,y)
  0.7291574286311803
  ```

Note: If only two classes are there, better use binarycrossentropy(ŷ, y) function.
"""
crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight=nothing) = _crossentropy(ŷ, y, weight)


function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* logsoftmax(logŷ) .* weight) * 1 // size(y, 2)
end

"""
    binarycrossentropy(ŷ, y; ϵ=eps(ŷ))

Return `-y*log(ŷ + ϵ) - (1-y)*log(1-ŷ + ϵ)`. The ϵ term provides numerical stability.

    julia> binarycrossentropy.(σ.([-1.1491, 0.8619, 0.3127]), [1, 1, 0.])
    3-element Array{Float64,1}:
    1.4244
    0.352317
    0.86167
"""
binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

# Re-definition to fix interaction with CuArrays.
CuArrays.@cufunc binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

"""
    logitbinarycrossentropy(logŷ, y)

`logitbinarycrossentropy(logŷ, y)` is mathematically equivalent to `binarycrossentropy(σ(logŷ), y)`
but it is more numerically stable.

    julia> logitbinarycrossentropy.([-1.1491, 0.8619, 0.3127], [1, 1, 0.])
    3-element Array{Float64,1}:
     1.4244
     0.352317
     0.86167
"""
logitbinarycrossentropy(logŷ, y) = (1 - y)*logŷ - logσ(logŷ)

# Re-definition to fix interaction with CuArrays.
CuArrays.@cufunc logitbinarycrossentropy(logŷ, y) = (1 - y)*logŷ - logσ(logŷ)

"""
    normalise(x::AbstractArray; dims=1)

Normalises `x` to mean 0 and standard deviation 1, across the dimensions given by `dims`. Defaults to normalising over columns.

    julia> a = reshape(collect(1:9), 3, 3)
    3×3 Array{Int64,2}:
     1  4  7
     2  5  8
     3  6  9

    julia> normalise(a)
    3×3 Array{Float64,2}:
     -1.22474  -1.22474  -1.22474
      0.0       0.0       0.0
      1.22474   1.22474   1.22474

    julia> normalise(a, dims=2)
    3×3 Array{Float64,2}:
     -1.22474  0.0  1.22474
     -1.22474  0.0  1.22474
     -1.22474  0.0  1.22474
"""
function normalise(x::AbstractArray; dims=1)
  μ′ = mean(x, dims = dims)
  σ′ = std(x, dims = dims, mean = μ′, corrected=false)
  return (x .- μ′) ./ σ′
end

"""
    kldivergence(ŷ, y)
KLDivergence is a measure of how much one probability distribution is different from the other.
It is always non-negative and zero only when both the distributions are equal everywhere.
[KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
"""
function kldivergence(ŷ, y)
  entropy = sum(y .* log.(y)) *1 //size(y,2)
  cross_entropy = crossentropy(ŷ, y)
  return entropy + cross_entropy
end

"""
    poisson(ŷ, y)
Poisson loss function is a measure of how the predicted distribution diverges from the expected distribution.
[Poisson Loss](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson).
"""
poisson(ŷ, y) = sum(ŷ .- y .* log.(ŷ)) *1 // size(y,2)

"""
    hinge(ŷ, y)

L1 loss function. Measures the loss given the prediction ŷ and true labels y(containing 1 or -1). 
[Hinge Loss](https://en.wikipedia.org/wiki/Hinge_loss).
"""
hinge(ŷ, y) = sum(max.(0, 1 .-  ŷ .* y)) *1 // size(y,2)

"""
    squared_hinge(ŷ, y)

L2 loss function. Computes squared hinge loss over the prediction ŷ and true labels y(conatining 1 or -1)
"""
squared_hinge(ŷ, y) = sum((max.(0,1 .-ŷ .* y)).^2) *1//size(y,2)

"""
    dice_coeff_loss(y_pred,y_true,smooth = 1)

Loss function used in Image Segmentation. Calculates loss based on dice coefficient. Similar to F1_score
    Dice_Coefficient(A,B) = 2*sum(|A*B|+smooth)/(sum(A^2)+sum(B^2)+ smooth)
    Dice_loss = 1-Dice_Coefficient

Ref: [V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation](https://arxiv.org/pdf/1606.04797v1.pdf)
"""
function dice_coeff_loss(y_pred,y_true,smooth=eltype(y_pred)(1.0))
    intersection = sum(y_true.*y_pred)
    return 1 - (2*intersection + smooth)/(sum(y_true.^2) + sum(y_pred.^2)+smooth)
end

"""
    tversky_loss(y_pred,y_true,beta = 0.7)

Used with imbalanced data to give more weightage to False negatives. Larger β weigh recall higher than precision (by placing more emphasis on false negatives)
    tversky_loss(ŷ,y,beta) = 1 - sum(|y.*ŷ| + 1) / (sum(y.*ŷ + beta*(1 .- y).*ŷ + (1 .- beta)*y.*(1 .- ŷ))+ 1)
Ref: [Tversky loss function for image segmentation using 3D fully convolutional deep networks](https://arxiv.org/pdf/1706.05721.pdf)
"""
function tversky_loss(y_pred,y_true,beta = eltype(y_pred)(0.7))
    intersection = sum(y_true.*y_pred)
    return 1 - (intersection+1)/(sum(y_true.*y_pred + beta*(1 .- y_true).* y_pred + (1-beta).*y_true.*(1 .- y_pred))+1)
end
