export mse, mse!

mse(ŷ, y) = sumabs2(ŷ .- y)/2

back!(::typeof(mse), Δ, ŷ, y) = Δ*(ŷ .- y)
