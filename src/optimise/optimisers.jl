struct SGD
  ps::Vector{Param}
  η::Float32
end

sgd(m, η) = SGD(params(m), η)

function update!(o::SGD)
  for p in o.ps
    p.x .-= p.Δ .* o.η
    Δ .= 0
  end
end
