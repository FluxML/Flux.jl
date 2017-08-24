struct SGD
  ps::Vector{Any}
  η::Float32
end

sgd(m, η) = SGD(params(m), η)

function update!(o::SGD)
  for p in o.ps
    x, Δ = data(p), grad(p)
    x .-= Δ .* o.η
    Δ .= 0
  end
end
