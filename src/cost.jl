export mse, mse!

function mse!(∇, pred, target)
  map!(-, ∇, pred, target)
  sumabs2(∇)/2
end

mse(pred, target) = mse(similar(pred), pred, target)
