export AArray

const AArray = AbstractArray

initn(dims...) = randn(dims...)/100

function train!(m, train, test = []; epoch = 1, batch = 10, η = 0.1)
    i = 0
    Δ = zeros(length(train[1][2]))
    for _ in 1:epoch
      @progress for (x, y) in train
        i += 1
        pred = m(x)
        any(isnan, pred) && error("NaN")
        err = mse!(Δ, pred, y)
        back!(m, Δ, x)
        i % batch == 0 && update!(m, η)
        i % 1000 == 0 && @show accuracy(m, test)
      end
    end
    return m
end

function accuracy(m, data)
  correct = 0
  for (x, y) in data
    onecold(m(x)) == onecold(y) && (correct += 1)
  end
  return correct/length(data)
end
