export AArray, onehot, onecold

const AArray = AbstractArray

onehot(label, labels) = [i == label for i in labels]
onecold(pred, labels = 1:length(pred)) = labels[findfirst(pred, maximum(pred))]

function train!(m::Model, train, test = []; epoch = 1, batch = 10, η = 0.1)
    i = 0
    ∇ = zeros(length(train[1][2]))
    for _ in 1:epoch
      for (x, y) in shuffle!(train)
        i += 1
        err = mse!(∇, m(x), y)
        back!(m, ∇)
        i % batch == 0 && update!(m, η/batch)
      end
      @show accuracy(m, test)
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
