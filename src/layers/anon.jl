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
