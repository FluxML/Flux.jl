tobatch(xs::Batch) = rawbatch(xs)
tobatch(xs) = tobatch(batchone(xs))

function accuracy(m, data)
  correct = 0
  for (x, y) in data
    x, y = tobatch.((x, y))
    correct += sum(onecold(m(x)) .== onecold(y))
  end
  return correct/length(data)
end

function train!(m, train, test = [];
                epoch = 1, η = 0.1, loss = mse)
    i = 0
    for e in 1:epoch
      info("Epoch $e")
      @progress for (x, y) in train
        x, y = tobatch.((x, y))
        i += 1
        ŷ = m(x)
        any(isnan, ŷ) && error("NaN")
        Δ = back!(loss, 1, ŷ, y)
        back!(m, Δ, x)
        update!(m, η)
        i % 1000 == 0 && @show accuracy(m, test)
      end
    end
    return m
end
