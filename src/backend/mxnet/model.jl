type MXModel
  model::Any
  params::Dict{Symbol,Any}
  exec::mx.Executor
end

mxdims(dims::NTuple) =
  length(dims) == 1 ? (1, dims...) : reverse(dims)

mxdims(n::Integer) = mxdims((n,))

function mxargs(args)
  map(args) do kv
    arg, value = kv
    arg => mx.zeros(mxdims(size(value)))
  end
end

function load!(model::MXModel)
  for (name, arr) in model.exec.arg_dict
    # TODO: don't allocate here
    haskey(model.params, name) && mx.copy_ignore_shape!(arr, model.params[name]')
  end
  return model
end

function mxnet(model::Model, input)
  vars = Dict{Symbol,Any}()
  node = graph(vars, model, mx.Variable(:input))
  args = merge(mxargs(vars), Dict(:input => mx.zeros(mxdims(input))))
  model = MXModel(model, vars, mx.bind(node, args = args, grad_req = mx.GRAD_NOP))
  load!(model)
  return model
end

function (model::MXModel)(input)
  inputnd = model.exec.arg_dict[:input]
  mx.copy_ignore_shape!(inputnd, input')
  mx.forward(model.exec)
  copy(model.exec.outputs[1])'
end
