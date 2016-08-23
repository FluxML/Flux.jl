export Model, back!, update!, param

# Basic model API

abstract Model

back!(m::Model, ∇) = error("Backprop not implemented for $(typeof(m))")
update!(m::Model, η) = m

# Model parameters

type Param{T}
  x::T
  Δx::T
end

param(x) = Param(x, zero(x))

state(p::Param) = p.x

function accumulate!(p::Param, Δ)
  p.Δx .+= Δ
  return p
end

function update!(p::Param, η)
  p.x .+= p.Δx .* η
  p.Δx[:] = 0
  return p
end

state(x) = x
accumulate!(x, Δ) = x

# Anonymous models

export Capacitor

type Capacitor <: Model
  forward::Function
  backward::Function
  update::Function
  graph::IVertex{Any}
end

(cap::Capacitor)(args...) = cap.forward(args...)

back!(cap::Capacitor, args...) = cap.backward(args...)

update!(cap::Capacitor, η) = cap.update(η)

graph(cap::Capacitor) = cap.graph
