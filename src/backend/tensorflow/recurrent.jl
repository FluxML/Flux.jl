immutable SeqModel
  m::Model
end

function tf(model::Flux.Unrolled)
  sess = Session(Graph())
  input = placeholder(Float32)
  inputs = TensorFlow.unpack(input, num = model.steps, axis = 1)
  params, (state, outputs) = tograph(model.graph, inputs...)
  output = TensorFlow.pack(outputs, axis = 1)
  run(sess, initialize_all_variables())
  Model(model, sess, params,
        [input], [output],
        [gradients(output, input)]) |> SeqModel
end

function batchseq(xs)
  dims = ndims(xs)-2
  T = Array{eltype(xs),dims}
  S = Array{eltype(xs),dims+1}
  B = Array{eltype(xs),dims+2}
  Batch{Seq{T,S},B}(xs)
end

(m::SeqModel)(x::BatchSeq) = batchseq(rawbatch(m.m(x)))

(m::SeqModel)(x::Seq) = first(m(batchone(x)))
