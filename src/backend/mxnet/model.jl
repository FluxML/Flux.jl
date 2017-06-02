using Flux: collectt, shapecheckt

function copyargs!(as, bs)
  for id in intersect(keys(as), keys(bs))
    copy!(as[id], bs[id])
  end
end

struct Graph
  input
  output
  params::Dict{Symbol,Any}
  stacks::Dict{Any,Any}
end

function mxparams(ps)
  params = Dict{Symbol,MXArray}()
  for (name, param) in ps
    params[name] = MXArray(size(param))
  end
  return params
end

ndparams(d) = Dict{Symbol,mx.NDArray}(k => v.data for (k, v) in d)

struct Exec
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

dictt(xs, ys) = Dict(zip(collectt(xs), collectt(ys)))

function executor(graph::Graph, input...)
  shapecheckt(graph.input, input)
  args  = merge(mxparams(graph.params), dictt(graph.input, mapt(d->MXArray(size(d)), input)))
  grads = filter((a, b) -> b isa Flux.Param, graph.params)
  grads = merge(mxparams(grads), dictt(graph.input, mapt(d->MXArray(size(d)), input)))
  exec = mx.bind(mxgroup(graph.output),
                 args = ndparams(args),
                 args_grad = ndparams(grads),
                 grad_req = mx.GRAD_ADD)
  exec = Exec(graph, exec, args, grads, MXArray.(exec.outputs))
  loadparams!(exec)
  return exec
end

function (exec::Exec)(input...)
  foreach(kv -> copy!(exec.args[kv[1]], kv[2]), dictt(exec.graph.input, input))
  mx.forward(exec.exec, is_train = true)
  mxungroup(exec.graph.output, copy(exec.outs))
end

function Flux.back!(exec::Exec, Δ)
  mapt(k -> exec.grads[k][:] = 0, exec.graph.input)
  mx.backward(exec.exec, map(x -> MXArray(x).data, collectt(Δ)))
  mapt(k -> copy(exec.grads[k]), exec.graph.input)
end

function Flux.update!(exec::Exec, η)
  for (arg, grad) in zip(exec.exec.arg_arrays, exec.exec.grad_arrays)
    grad == nothing && continue
    mx.@nd_as_jl rw = (arg, grad) begin
      arg .-= grad .* η
      grad[:] = 0
    end
  end
  storeparams!(exec)
  return exec
end

# TODO: if `last` changes, update params appropriately

mutable struct Model <: Flux.Model
  model::Any
  execs::Dict{Tuple,Exec}
  graph::Graph
  last::Exec
  Model(model) = new(model, Dict())
end

mxnet(model) = Model(model)

import Base: @get!

# TODO: dims having its own type would be useful
executor(m::Model, input...) = @get!(m.execs, mapt(size, input), executor(m.graph, input...))

function (m::Model)(xs...)
  @mxerr m.graph.stacks begin
    !isdefined(m, :graph) &&
      (m.graph = tograph(m.model, mapt(_ -> gensym("input"), xs)...))
    m.last = exec = executor(m, xs...)
    exec(xs...)
  end
end

function Flux.back!(m::Model, Δ, xs...)
  m.last = exec = m.execs[mapt(size, xs)]
  back!(exec, Δ)
end

Flux.update!(m::Model, η) = (update!(m.last, η); m)

# Recurrent Models

using Flux: Stateful, SeqModel

mxnet(m::Stateful) = Stateful(mxnet(m.model), m.istate, m.ostate)
mxnet(m::SeqModel) = SeqModel(mxnet(m.model), m.steps)

# MX FeedForward interface

struct SoftmaxOutput
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
  graph = tograph(model, input, feedforward=true)
  ff = mx.FeedForward(graph.output, context = context)
  isempty(graph.params) || (ff.arg_params = ndparams(mxparams(graph.params)))
  return ff
end
