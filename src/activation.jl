export Sigmoid

σ(x) = 1/(1+exp(-x))
σ′(x) = σ(x)*(1-σ(x))

∇₁(::typeof(σ)) = σ′

type Sigmoid <: Activation
  in::Vector{Float32}
  out::Vector{Float32}
  ∇in::Vector{Float32}
end

Sigmoid(size::Integer) = Sigmoid(zeros(size), zeros(size), zeros(size))

function (l::Sigmoid)(x)
  l.in = x
  map!(σ, l.out, x)
end

function back!(l::Sigmoid, ∇)
  map!(σ′, l.∇in, l.in)
  map!(*, l.∇in, l.∇in, ∇)
end

shape(l::Sigmoid) = length(l.in)

Sigmoid() = Init(in -> Sigmoid(in[1]))
