export mse, mse!

function mse!(Δ, pred, target)
  map!(-, Δ, pred, target)
  sumabs2(Δ)/2
end

mse(pred, target) = mse(similar(pred), pred, target)
