using MacroTools

type MXModel <: Model
  model::Any
  params::Dict{Symbol,Any}
  grads::Dict{Symbol,Any}
  exec::mx.Executor
end

Base.show(io::IO, m::MXModel) =
  print(io, "MXModel($(m.model))")

mxdims(dims::NTuple) =
  length(dims) == 1 ? (1, dims...) : reverse(dims)

mxdims(n::Integer) = mxdims((n,))

function tond!(nd::mx.NDArray, xs::AArray)
  mx.copy_ignore_shape!(nd, xs')
  nd
end

tond(xs::AArray) = tond!(mx.zeros(mxdims(size(xs))), xs)

fromnd(xs::mx.NDArray) = copy(xs)'

ndzero!(xs::mx.NDArray) = copy!(xs, mx.zeros(size(xs)))

function mxargs(args)
  map(args) do kv
    arg, value = kv
    arg => mx.zeros(mxdims(size(value)))
  end
end

function mxgrads(mxargs)
  map(mxargs) do kv
    arg, value = kv
    arg => mx.zeros(size(value))
  end
end

function load!(model::MXModel)
  for (name, arr) in model.exec.arg_dict
    haskey(model.params, name) && tond!(arr, model.params[name])
  end
  return model
end

function mxnet(model::Model, input)
  vars = Dict{Symbol,Any}()
  node = graph(vars, model, mx.Variable(:input))
  args = merge(mxargs(vars), Dict(:input => mx.zeros(mxdims(input))))
  grads = mxgrads(args)
  model = MXModel(model, vars, grads,
                  mx.bind(node, args = args,
                                args_grad = grads,
                                grad_req = mx.GRAD_ADD))
  load!(model)
  return model
end

function (model::MXModel)(input)
  tond!(model.exec.arg_dict[:input], input)
  mx.forward(model.exec, is_train = true)
  fromnd(model.exec.outputs[1])
end

function Flux.back!(model::MXModel, Δ, x)
  ndzero!(model.grads[:input])
  mx.backward(model.exec, tond(Δ))
  fromnd(model.grads[:input])
end

function Flux.update!(model::MXModel, η)
  for (arg, grad) in zip(model.exec.arg_arrays, model.exec.grad_arrays)
    mx.@nd_as_jl rw = (arg, grad) begin
      arg .-= grad .* η
      grad[:] = 0
    end
  end
  return model
end
