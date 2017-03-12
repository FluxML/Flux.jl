using Flux: runrawbatched

type AlterParam
  param
  load
  store
end

Base.size(p::AlterParam) = size(p.load(p.param.x))
Base.copy!(xs, p::AlterParam) = copy!(xs, p.load(p.param.x))

function copyargs!(as, bs)
  for id in intersect(keys(as), keys(bs))
    copy!(as[id], bs[id])
  end
end

type Graph
  output
  params::Dict{Symbol,Any}
  stacks::Dict{Any,Any}
end

function mxparams(g::Graph)
  params = Dict{Symbol,MXArray}()
  for (name, param) in g.params
    params[name] = MXArray(size(param))
  end
  return params
end

ndparams(d::Dict{Symbol,MXArray}) = Dict(k => v.data for (k, v) in d)

type Exec <: Flux.Model
  graph::Graph
  exec::mx.Executor
  args::Dict{Symbol,MXArray}
  grads::Dict{Symbol,MXArray}
  outs::Vector{MXArray}
end

loadparams!(exec::Exec) = copyargs!(exec.args, exec.graph.params)
storeparams!(exec::Exec) = copyargs!(exec.graph.params, exec.args)

mxgroup(x) = x
mxgroup(x::Tuple) = mx.Group(mxgroup.(x)...)
mxungroup(x, outs) = copy(shift!(outs))
mxungroup(x::Tuple, outs) = map(x -> mxungroup(x, outs), x)

function executor(graph::Graph, input)
  args  = merge(mxparams(graph), Dict(:input => MXArray(input)))
  grads = merge(mxparams(graph), Dict(:input => MXArray(input)))
  exec = mx.bind(mxgroup(graph.output),
                 args = ndparams(args),
                 args_grad = ndparams(grads),
                 grad_req = mx.GRAD_ADD)
  exec = Exec(graph, exec, args, grads, MXArray.(exec.outputs))
  loadparams!(exec)
  return exec
end

function (exec::Exec)(input)
  copy!(exec.args[:input], input)
  mx.forward(exec.exec, is_train = true)
  mxungroup(exec.graph.output, copy(exec.outs))
end

function Flux.back!(exec::Exec, Δ)
  exec.grads[:input][:] = 0
  mx.backward(exec.exec, MXArray(Δ).data)
  copy(exec.grads[:input])
end

function Flux.update!(exec::Exec, η)
  for (arg, grad) in zip(exec.exec.arg_arrays, exec.exec.grad_arrays)
    mx.@nd_as_jl rw = (arg, grad) begin
      arg .-= grad .* η
      grad[:] = 0
    end
  end
  storeparams!(exec)
  return exec
end

# TODO: if `last` changes, update params appropriately

type Model
  model::Any
  graph::Graph
  execs::Dict{Tuple,Exec}
  last::Exec
  Model(model, graph, execs) = new(model, graph, execs)
end

function mxnet(model)
  graph = tograph(model, mx.Variable(:input))
  Model(model, graph, Dict())
end

import Base: @get!

executor(m::Model, input) = @get!(m.execs, input, executor(m.graph, input))

function (m::Model)(x)
  runrawbatched(x) do x
    m.last = exec = @mxerr m.graph.stacks executor(m, size(x))
    exec(x)
  end
end

function Flux.back!(m::Model, Δ, x)
  runrawbatched(Δ, x) do Δ, x
    m.last = exec = m.execs[size(x)]
    back!(exec, Δ)
  end
end

Flux.update!(m::Model, η) = (update!(m.last, η); m)

# MX FeedForward interface

type SoftmaxOutput
  name::Symbol
end

graph(s::SoftmaxOutput, xs) = mx.SoftmaxOutput(xs, name = s.name)

function rewrite_softmax(model, name)
  model == softmax && return SoftmaxOutput(name)
  g = Flux.graph(model)
  (g == nothing || g.value ≠ softmax || DataFlow.nin(g) ≠ 1) && error("mx.FeedForward models must end with `softmax`")
  return Flux.Capacitor(vertex(SoftmaxOutput(name), g[1]))
end

function mx.FeedForward(model::Flux.Model; input = :data, label = :softmax, context = mx.cpu())
  model = rewrite_softmax(model, label)
  graph = tograph(model, mx.Variable(input), feedforward=true)
  ff = mx.FeedForward(graph.output, context = context)
  isempty(graph.params) || (ff.arg_params = ndparams(mxparams(graph)))
  return ff
end
