using Flux: batchone, unbatchone, rebatch

# MNet batches on last dimension
rebatch_last(xs) = permutedims(xs, (2:ndims(xs)..., 1))
rebatch_first(xs) = permutedims(xs, (ndims(xs), 1:ndims(xs)-1...))
rebatch_first(xs::Tuple) = rebatch_first.(xs)

paramvalue(p) = rebatch_last(p)
paramvalue(p::Flux.Param) = paramvalue(p.x)

# Basically a kludge to make Affine work
# Hopefully will go away with more inference
type AlterParam
  param::Flux.Param
  strip::Bool
  rebatch::Bool
end

function paramvalue(p::AlterParam)
  val = p.rebatch ? paramvalue(p.param) : p.param.x
  p.strip ? squeeze(val, 1) : val
end

type Graph
  output
  params::Dict{Symbol,Any}
  stacks::Dict{Any,Any}
end

function mxparams(g::Graph)
  params = Dict{Symbol,mx.NDArray}()
  for (name, param) in g.params
    params[name] = mx.zeros(size(paramvalue(param)))
  end
  return params
end

function loadparams!(g::Graph, args)
  for (id, param) in g.params
    haskey(args, id) && copy!(args[id], paramvalue(param))
  end
end

function storeparams!(g::Graph, args)
  for (id, param) in g.params
    haskey(args, id) && copy!(param.x, rebatch_first(copy(args[id])))
  end
end

type Model <: Flux.Model
  model::Any
  graph::Graph
  grads::Dict{Symbol,Any}
  exec::mx.Executor
end

loadparams!(model::Model) = loadparams!(model.graph, model.exec.arg_dict)
storeparams!(model::Model) = storeparams!(model.graph, model.exec.arg_dict)

mxgroup(x) = x
mxgroup(x::Tuple) = mx.Group(mxgroup.(x)...)
mxungroup(x, outs) = copy(shift!(outs))
mxungroup(x::Tuple, outs) = map(x -> mxungroup(x, outs), x)

function mxnet(model::Flux.Model, input)
  graph = tograph(model, mx.Variable(:input))
  args  = merge(mxparams(graph), Dict(:input => mx.zeros(input)))
  grads = merge(mxparams(graph), Dict(:input => mx.zeros(input)))
  model = @mxerr graph.stacks Model(model, graph, grads,
                                    mx.bind(mxgroup(graph.output), args = args,
                                            args_grad = grads,
                                            grad_req = mx.GRAD_ADD))
  loadparams!(model)
  return model
end

function runmodel(model::Model, input)
  copy!(model.exec.arg_dict[:input], input)
  mx.forward(model.exec, is_train = true)
  mxungroup(model.graph.output, copy(model.exec.outputs))
end

(m::Model)(x::Batch) = rebatch(rebatch_first(runmodel(m, rebatch_last(rawbatch(x)))))

(m::Model)(x) = unbatchone(m(batchone(x)))

tond(xs::AArray) = copy!(mx.zeros(size(xs)), xs)

function runback!(model::Model, Δ)
  model.grads[:input][:] = 0
  mx.backward(model.exec, tond(Δ))
  copy(model.grads[:input])
end

Flux.back!(m::Model, Δ::Batch, x) = rebatch(rebatch_first(runback!(m, rebatch_last(rawbatch(Δ)))))

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
  isempty(graph.params) || (ff.arg_params = mxparams(graph))
  return ff
end
