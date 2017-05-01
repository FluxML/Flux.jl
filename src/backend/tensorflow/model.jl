using Flux: mapt

struct Exec
  session::Session
  input::Any
  output::Any
  params::Dict{Flux.Param,Tensor}
  stacks::Dict{Any,Any}
end

function makesession(model, inputs; session = Session(Graph()))
  params, stacks, output = tograph(model, inputs...)
  run(session, global_variables_initializer())
  Exec(session, inputs, output, params, stacks)
end

function makesession(model, n::Integer; session = Session(Graph()))
  makesession(model, [placeholder(Float32) for _ = 1:n], session = session)
end

retuple(xs) = xs
retuple(xs::AbstractArray{<:AbstractArray}) = (retuple.(xs)...,)

function (m::Exec)(args...)
  @assert length(args) == length(m.input)
  retuple(run(m.session, m.output, Dict(zip(m.input, args))))
end

mutable struct Model
  model::Any
  exec::Exec
  Model(model) = new(model)
end

tf(model) = Model(model)

function (m::Model)(args...)
  args = mapt(x->convert.(Float32, x),args)
  isdefined(m, :graph) || (m.exec = makesession(m.model, length(args)))
  @tferr m.exec.stacks m.exec(args...)
end

for f in :[back!, update!].args
  @eval function Flux.$f(m::Model, args...)
    error($(string(f)) * " is not yet supported on TensorFlow models")
  end
end
