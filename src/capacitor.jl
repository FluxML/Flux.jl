type Capacitor{T}
  Δs::Vector{T}
end

type Patch{T}
  η::Float32
  Δs::Capacitor{T}
end
