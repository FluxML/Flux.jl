module TF

using ..Flux, Flow, TensorFlow

# Workaround for tensor display bug
using Juno
Media.render(::Juno.Clipboard, ::Tensor) = "Tensor()"

cvalue(x) = x
cvalue(c::Constant) = c.value
cvalue(v::Vertex) = cvalue(value(v))

graph(x::Tensor) = x

matrixify(xs) = xs
matrixify(xs::Vector) = xs[:,1:1]
# TODO: detect variable reuse
graph{T<:AArray}(p::Flux.Param{T}) = Variable(matrixify(p.x))

function graph(model::Model, args...)
  g = Flux.graph(model)
  g = Flow.mapconst(g) do x
    !isa(x, Flux.ModelInput) ? x :
    isa(x.name, Integer) ? args[x.name] : getfield(model, x.name)
  end
  postwalk(g) do v
    vertex(graph(cvalue(v), cvalue.(inputs(v))...))
  end |> value
end

graph(::typeof(*), args...) = *(args...)
graph(::typeof(+), args...) = +(args...)

type Model
  session::Session
  inputs::Vector{Tensor}
  graph::Tensor
  grad::Tensor
end

Media.render(::Juno.Clipboard, ::Model) = "Flux.TF.Model()"

function tf(model)
  sess = Session()
  input = placeholder(Float64)
  g = graph(model, input)
  run(sess, initialize_all_variables())
  Model(sess, [input], g, gradients(g, input))
end

function (m::Model)(args...)
  @assert length(args) == length(m.inputs)
  run(m.session, m.graph, Dict(zip(m.inputs, args)))
end

function Flux.back!(m::Model, Δ, args...)
  @assert length(args) == length(m.inputs)
  # TODO: keyword arguments to `gradients`
  run(m.session, m.grad, Dict(zip(m.inputs, args)))
end

function Flux.update!(m::Model)
  error("update! is not yet supported on TensorFlow models")
end

end
