using Flux: mapt, collectt, shapecheckt

struct Exec
  session ::Session
  input   ::Any
  output  ::Any
  params  ::Dict{Param,Param{Tensor}}
  stacks  ::Dict{Any,Any}
end

function makesession(model, inputs; session = Session(Graph()))
  inputs = mapt(_ -> placeholder(Float32), inputs)
  params, stacks, output = tograph(model, inputs...)
  output = mapt(x->Param{Tensor}(x, placeholder(Float32)), output)
  params = Dict(x=>Param{Tensor}(y, gradients(mapt(x->x.x, output),
                                 y, mapt(x->x.Δx, output))) for (x, y) in params)
  inputs = mapt(x->Param{Tensor}(x, gradients(mapt(x->x.x, output),
                                 x, mapt(x->x.Δx, output))), inputs)
  run(session, global_variables_initializer())
  Exec(session, inputs, output, params, stacks)
end

retuple(xs) = xs
retuple(xs::AbstractArray{<:AbstractArray}) = (retuple.(xs)...,)

dictt(xs, ys) = Dict(zip(collectt(xs), collectt(ys)))

function Flux.params(m::Exec)
  collect(keys(m.params))
end

function (m::Exec)(args...)
  dict = merge(
    Dict(y.x=>x.x for (x, y) in m.params),
    Dict(x.x=>y for (x, y) in zip(m.input, args))
  )
  retuple(run(m.session, mapt(x->x.x, m.output), dict))
end

function Flux.back!(m::Exec, Δ, args...)
  dict = merge(
    Dict(y.x=>x.x for (x, y) in m.params),
    Dict(x.x=>y for (x, y) in zip(m.input, args)),
    Dict(x.Δx=>y for (x, y) in zip(collectt(m.output), collectt(Δ)))
  )    

  Δin, Δps = run(m.session, (mapt(x->x.Δx, m.input), map(x->x.Δx, values(m.params))), dict)

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

Flux.back!(m::Model, Δ, args...) = Flux.back!(m.exec, Δ, args...)
Flux.update!(m::Model, η) = (Flux.update!(m.exec, η); m)
Flux.params(m::Model) = Flux.params(m.exec)

# Recurrent Models

using Flux: Stateful, SeqModel

tf(m::Stateful) = Stateful(tf(m.model), m.states, m.istate, m.ostate)
tf(m::SeqModel) = SeqModel(tf(m.model), m.steps)
