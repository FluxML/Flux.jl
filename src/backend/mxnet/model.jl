using Flux: collectt, shapecheckt

struct AlterParam
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

struct Graph
  input
  output
  params::Dict{Symbol,Any}
  stacks::Dict{Any,Any}
end

mxparams(d) = Dict{Symbol,MXArray}(k => MXArray(size(v)) for (k, v) in d)
ndparams(d) = Dict{Symbol,mx.NDArray}(k => v.data for (k, v) in d)

struct Exec
  model::Model
  exec::mx.Executor
  args::Dict{Symbol,MXArray}
  grads::Dict{Symbol,MXArray}
  outs::Vector{MXArray}
end

loadparams!(m::Model)  = copyargs!(m.args, m.graph.params)
storeparams!(m::Model) = copyargs!(m.graph.params, m.args)

mxgroup(x) = x
mxgroup(x::Tuple) = mx.Group(mxgroup.(x)...)
mxungroup(x, outs) = copy(shift!(outs))
mxungroup(x::Tuple, outs) = map(x -> mxungroup(x, outs), x)

dictt(xs, ys) = Dict(zip(collectt(xs), collectt(ys)))

function (exec::Exec)(input...)
  foreach(kv -> copy!(exec.args[kv[1]], kv[2]), dictt(exec.model.graph.input, input))
  mx.forward(exec.exec, is_train = true)
  mxungroup(exec.model.graph.output, copy(exec.outs))
end

function Flux.back!(exec::Exec, Δ)
  mapt(k -> exec.grads[k][:] = 0, exec.model.graph.input)
  mx.backward(exec.exec, map(x -> MXArray(x).data, collectt(Δ)))
  mapt(k -> copy(exec.grads[k]), exec.model.graph.input)
end

function Flux.update!(m::Model, η)
  for param in keys(m.args)
    arg, grad = m.args[param].data, m.grads[param].data
    mx.@nd_as_jl rw = (arg, grad) begin
      arg .-= grad .* η
      grad[:] = 0
    end
  end
  storeparams!(m)
  return m
end

# TODO: if `last` changes, update params appropriately

mutable struct Model <: Flux.Model
  model::Any
  execs::Dict{Tuple,Exec}
  graph::Graph
  args::Dict{Symbol,MXArray}
  grads::Dict{Symbol,MXArray}
  Model(model) = new(model, Dict())
end

mxnet(model) = Model(model)

# TODO: dims having its own type would be useful

function executor(m::Model, input...)
  shapecheckt(m.graph.input, input)
  input_size = mapt(size, input)
  if input_size ∉ keys(m.execs)
    args  = merge(m.args,  dictt(m.graph.input, mapt(MXArray∘size, input)))
    grads = merge(m.grads, dictt(m.graph.input, mapt(MXArray∘size, input)))
    exec = mx.bind(mxgroup(m.graph.output),
                   args = ndparams(args),
                   args_grad = ndparams(grads),
                   grad_req = mx.GRAD_ADD)
    m.execs[input_size] = Exec(m, exec, args, grads, MXArray.(exec.outputs))
  end
  return m.execs[input_size]
end

function (m::Model)(xs...)
  @mxerr m.graph.stacks begin
    if !isdefined(m, :graph)
      m.graph = tograph(m.model, mapt(_ -> gensym("input"), xs)...)
      m.args  = mxparams(m.graph.params)
      m.grads = mxparams(m.graph.params)
      loadparams!(m)
    end
    executor(m, xs...)(xs...)
  end
end

Flux.back!(m::Model, Δ, xs...) =
  back!(m.execs[mapt(size, xs)], Δ)

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
