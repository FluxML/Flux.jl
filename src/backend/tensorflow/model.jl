using Flux: mapt, collectt, shapecheckt

struct Exec
  session::Session
  input::Any
  output::Any
  params::Dict{Flux.Param,Tensor}
  stacks::Dict{Any,Any}
end

function makesession(model, inputs; session = Session(Graph()))
  inputs = mapt(_ -> placeholder(Float32), inputs)
  params, stacks, output = tograph(model, inputs...)
  run(session, global_variables_initializer())
  Exec(session, inputs, output, params, stacks)
end

retuple(xs) = xs
retuple(xs::AbstractArray{<:AbstractArray}) = (retuple.(xs)...,)

dictt(xs, ys) = Dict(zip(collectt(xs), collectt(ys)))

function (m::Exec)(args...)
  shapecheckt(m.input, args)
  idict = dictt(m.input, args)
  pdict = Dict(t => p.x for (p, t) in m.params)
  retuple(run(m.session, m.output, merge(idict, pdict)))
end

mutable struct Model
  model::Any
  exec::Exec
  Model(model) = new(model)
end

tf(model) = Model(model)

function (m::Model)(args...)
  args = mapt(x->Float32.(x), args)
  isdefined(m, :graph) || (m.exec = makesession(m.model, args))
  @tferr m.exec.stacks m.exec(args...)
end

for f in :[back!, update!].args
  @eval function Flux.$f(m::Model, args...)
    error($(string(f)) * " is not yet supported on TensorFlow models")
  end
end

# Recurrent Models

using Flux: Stateful, SeqModel

tf(m::Stateful) = Stateful(tf(m.model), m.istate, m.ostate)
tf(m::SeqModel) = SeqModel(tf(m.model), m.steps)
