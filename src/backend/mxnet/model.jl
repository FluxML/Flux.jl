using Flux: batchone, unbatchone, rebatch

type AlterParam
  param
  load
  store
end

Base.size(p::AlterParam) = size(p.load(p.param.x))

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

function copyargs!(as, bs)
  for id in intersect(keys(as), keys(bs))
    copy!(as[id], bs[id])
  end
end

ndparams(d::Dict{Symbol,MXArray}) = Dict(k => v.data for (k, v) in d)

type Model <: Flux.Model
  model::Any
  graph::Graph
  args::Dict{Symbol,MXArray}
  grads::Dict{Symbol,MXArray}
  outs::Vector{MXArray}
  exec::mx.Executor
end

loadparams!(model::Model) = copyargs!(model.args, model.graph.params)
storeparams!(model::Model) = copyargs!(model.graph.params, model.args)

mxgroup(x) = x
mxgroup(x::Tuple) = mx.Group(mxgroup.(x)...)
mxungroup(x, outs) = copy(shift!(outs))
mxungroup(x::Tuple, outs) = map(x -> mxungroup(x, outs), x)

function mxnet(model::Flux.Model, input)
  graph = tograph(model, mx.Variable(:input))
  args  = merge(mxparams(graph), Dict(:input => MXArray(input)))
  grads = merge(mxparams(graph), Dict(:input => MXArray(input)))
  exec = @mxerr graph.stacks mx.bind(mxgroup(graph.output),
                                     args = ndparams(args),
                                     args_grad = ndparams(grads),
                                     grad_req = mx.GRAD_ADD)
  model = Model(model, graph, args, grads, MXArray.(exec.outputs), exec)
  loadparams!(model)
  return model
end

function runmodel(model::Model, input)
  copy!(model.args[:input], input)
  mx.forward(model.exec, is_train = true)
  mxungroup(model.graph.output, copy(model.outs))
end

(m::Model)(x::Batch) = rebatch(runmodel(m, rawbatch(x)))

(m::Model)(x) = unbatchone(m(batchone(x)))

function runback!(model::Model, Δ)
  model.grads[:input][:] = 0
  mx.backward(model.exec, MXArray(Δ).data)
  copy(model.grads[:input])
end

Flux.back!(m::Model, Δ::Batch, x) = rebatch(runback!(m, rawbatch(Δ)))

Flux.back!(m::Model, Δ, x) = first(Flux.back!(m, batchone(Δ), x))

function Flux.update!(model::Model, η)
  for (arg, grad) in zip(model.exec.arg_arrays, model.exec.grad_arrays)
    mx.@nd_as_jl rw = (arg, grad) begin
      arg .-= grad .* η
      grad[:] = 0
    end
  end
  storeparams!(model)
  return model
end

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
