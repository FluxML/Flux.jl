type Model
  model::Any
  session::Session
  params::Dict{Flux.Param,Tensor}
  inputs::Vector{Tensor}
  output::Any
  gradients::Vector{Tensor}
end

ismultioutput(m::Model) = !isa(m.output, Tensor)

function tf(model)
  sess = Session(Graph())
  input = placeholder(Float32)
  params, output = tograph(model, input)
  run(sess, initialize_all_variables())
  Model(model, sess, params,
        [input], output,
        [gradients(output, input)])
end

batchone(x) = Batch((x,))

function batch(xs)
  dims = ndims(xs)-1
  T = Array{eltype(xs),dims}
  B = Array{eltype(xs),dims+1}
  Batch{T,B}(xs)
end

function (m::Model)(args::Batch...)
  @assert length(args) == length(m.inputs)
  output = run(m.session, m.output, Dict(zip(m.inputs, args)))
  ismultioutput(m) ? (batch.(output)...,) : batch(output)
end

function (m::Model)(args...)
  output = m(map(batchone, args)...)
  ismultioutput(m) ? map(first, output) : first(output)
end

function Flux.back!(m::Model, Δ, args...)
  @assert length(args) == length(m.inputs)
  # TODO: keyword arguments to `gradients`
  run(m.session, m.gradients[1], Dict(zip(m.inputs, args)))
end

function Flux.update!(m::Model)
  error("update! is not yet supported on TensorFlow models")
end

import Juno: info

function Flux.train!(m::Model, train, test=[]; epoch = 1, η = 0.1,
                     loss = (y, y′) -> reduce_sum((y - y′).^2)/2,
                     opt = TensorFlow.train.GradientDescentOptimizer(η))
  i = 0
  Y = placeholder(Float32)
  Loss = loss(m.outputs[1], Y)
  minimize_op = TensorFlow.train.minimize(opt, Loss)
  for e in 1:epoch
    info("Epoch $e\n")
    @progress for (x, y) in train
      y, cur_loss, _ = run(m.session, vcat(m.outputs[1], Loss, minimize_op),
                           Dict(m.inputs[1]=>batchone(x), Y=>batchone(y)))
      if i % 5000 == 0
        @show y
        @show accuracy(m, test)
      end
      i += 1
    end
  end
end
