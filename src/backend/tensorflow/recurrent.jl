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
    [])
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
