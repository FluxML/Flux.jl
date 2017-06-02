using Flux: mapt, collectt, shapecheckt

struct Exec
  session ::Session
  input   ::Any
  output  ::Any
  grads   ::Any
  params  ::Dict{Flux.Param,Tensor}
  stacks  ::Dict{Any,Any}
end

function makesession(model, inputs; session = Session(Graph()))
  inputs = mapt(_ -> placeholder(Float32), inputs)
  params, stacks, output = tograph(model, inputs...)
  # grads = gradients(output, [collectt(inputs)..., values(params)...])
  grads = placeholder(Float32)
  run(session, global_variables_initializer())
  Exec(session, inputs, output, grads, params, stacks)
end

retuple(xs) = xs
retuple(xs::AbstractArray{<:AbstractArray}) = (retuple.(xs)...,)

dictt(xs, ys) = Dict(zip(collectt(xs), collectt(ys)))

function params(m::Exec, args...)
  shapecheckt(m.input, args)
  idict = dictt(m.input, args)
  pdict = Dict(t => p.x for (p, t) in m.params)
  merge(idict, pdict)
end

function (m::Exec)(args...)
  retuple(run(m.session, m.output, params(m, args...)))
end

pullt!(_, xs) = shift!(xs)
pullt!(x::Tuple, xs) = map(x -> pullt!(x, xs), x)

# TODO: gradients don't work yet
# `gradients` lacks support for `grad_y`s and multiple `y`s

function Flux.back!(m::Exec, Δ, args...)
  Δps = run(m.session, m.grads, params(m, args...))
  Δin = pullt!(m.input, Δps)
  for (p, Δ) in zip(keys(m.params), Δps)
    p.Δx .+= Δ
  end
  Δin
end

function Flux.update!(m::Exec, η)
  for p in keys(m.params)
    update!(p, η)
  end
  return m
end

mutable struct Model
  model::Any
  exec::Exec
  Model(model) = new(model)
end

tf(model) = Model(model)

function (m::Model)(args...)
  args = mapt(x->Float32.(x), args)
  isdefined(m, :exec) || (m.exec = makesession(m.model, args))
  @tferr m.exec.stacks m.exec(args...)
end

Flux.back!(m::Model, Δ, args...) = back!(m.exec, Δ, args...)
Flux.update!(m::Model, η) = (update!(m.exec, η); m)

# Recurrent Models

using Flux: Stateful, SeqModel

tf(m::Stateful) = Stateful(tf(m.model), m.states, m.istate, m.ostate)
tf(m::SeqModel) = SeqModel(tf(m.model), m.steps)
