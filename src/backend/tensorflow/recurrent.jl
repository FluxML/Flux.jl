# TODO: refactor, some of this is more general than just the TF backend

type SeqModel
  m::Model
  state::Any
end

cgroup(xs...) = Flow.group(map(constant, xs)...)

function tf(model::Flux.Unrolled)
  sess = Session(Graph())
  input = placeholder(Float32)
  instates = [placeholder(Float32) for _ in model.states]
  inputs = TensorFlow.unpack(input, num = model.steps, axis = 1)
  params, (outstates, outputs) = tograph(model.graph, cgroup(instates...), cgroup(inputs...))
  output = TensorFlow.pack(outputs, axis = 1)
  run(sess, initialize_all_variables())
  SeqModel(
    Model(model, sess, params,
          [instates..., input], [outstates..., output],
          [gradients(output, input)]),
    batchone.(model.states))
end

function batchseq(xs)
  dims = ndims(xs)-2
  T = Array{eltype(xs),dims}
  S = Array{eltype(xs),dims+1}
  B = Array{eltype(xs),dims+2}
  Batch{Seq{T,S},B}(xs)
end

function (m::SeqModel)(x::BatchSeq)
  if isempty(m.state) || length(first(m.state)) ≠ length(x)
    m.state = map(batchone, m.m.model.states)
  end
  output = m.m(m.state..., x)
  m.state, output = output[1:end-1], output[end]
  return batchseq(rawbatch(output))
end

(m::SeqModel)(x::Seq) = first(m(batchone(x)))

function Flux.train!(m::SeqModel, train; epoch = 1, η = 0.1,
                     loss = (y, y′) -> reduce_sum((y - y′).^2)/2,
                     opt = TensorFlow.train.GradientDescentOptimizer(η))
  Y = placeholder(Float32)
  Loss = loss(m.m.output[end], Y)
  minimize_op = TensorFlow.train.minimize(opt, Loss)
  for e in 1:epoch
    info("Epoch $e\n")
    @progress for (x, y) in train
      y, cur_loss, _ = run(m.m.session, vcat(m.m.output[end], Loss, minimize_op),
                           merge(Dict(m.m.inputs[end]=>x, Y=>y),
                                 Dict(zip(m.m.inputs[1:end-1], m.state))))
    end
  end
end
