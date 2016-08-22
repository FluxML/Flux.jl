type MXModel
  model::Any
  params::Dict{Symbol,Any}
  exec::mx.Executor
end

mxdims(dims::NTuple) =
  length(dims) == 1 ? (1, dims...) : reverse(dims)

function mxargs(args)
  map(args) do kv
    arg, value = kv
    arg => mx.zeros(mxdims(size(value)))
  end
end

function mxnet(model::Model, input)
  vars = Dict{Symbol,Any}(:input => mx.zeros(mxdims(input)))
  node = graph(vars, model, mx.Variable(:input))
  MXModel(model, vars, mx.bind(node, args = mxargs(vars), grad_req = mx.GRAD_NOP))
end
