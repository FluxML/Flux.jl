type Model
  session::Session
  inputs::Vector{Tensor}
  graph::Tensor
  grad::Tensor
end

function tf(model)
  sess = Session(Graph())
  input = placeholder(Float32)
  g = graph(model, input)
  run(sess, initialize_all_variables())
  Model(sess, [input], g, gradients(g, input))
end

batch(x) = Batch((x,))

function (m::Model)(args::Batch...)
  @assert length(args) == length(m.inputs)
  run(m.session, m.graph, Dict(zip(m.inputs, args)))
end

(m::Model)(args...) = m(map(batch, args)...)

function Flux.back!(m::Model, Δ, args...)
  @assert length(args) == length(m.inputs)
  # TODO: keyword arguments to `gradients`
  run(m.session, m.grad, Dict(zip(m.inputs, args)))
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
  Loss = loss(m.graph, Y)
  minimize_op = TensorFlow.train.minimize(opt, Loss)
  for e in 1:epoch
    info("Epoch $e\n")
    @progress for (x, y) in train
      y, cur_loss, _ = run(m.session, vcat(m.graph, Loss, minimize_op),
                           Dict(m.inputs[1]=>batch(x), Y=>batch(y)))
      if i % 5000 == 0
        @show y
        @show accuracy(m, test)
      end
      i += 1
    end
  end
end
