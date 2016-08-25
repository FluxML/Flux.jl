export AArray, onehot, onecold

const AArray = AbstractArray

onehot(label, labels) = [i == label for i in labels]
onecold(pred, labels = 1:length(pred)) = labels[findfirst(pred, maximum(pred))]

initn(dims...) = randn(dims...)/100

function train!(m, train, test = []; epoch = 1, batch = 10, η = 0.1)
    i = 0
    ∇ = zeros(length(train[1][2]))
    for _ in 1:epoch
      for (x, y) in train
        i += 1
        pred = m(x)
        any(isnan, pred) && error("NaN")
        err = mse!(∇, pred, y)
        back!(m, ∇, x)
        i % batch == 0 && update!(m, η)
        i % 1000 == 0 && @show accuracy(m, test)
      end
    end
    return m
end

function accuracy(m::Model, data)
  correct = 0
  for (x, y) in data
    onecold(m(x)) == onecold(y) && (correct += 1)
  end
  return correct/length(data)
end
