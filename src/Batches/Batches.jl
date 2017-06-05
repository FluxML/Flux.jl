module Batches

using Juno, Lazy

export CatMat, rawbatch,
  Batch, Batched, batchone, tobatch, rebatch,
  Seq, BatchSeq, rebatchseq

include("catmat.jl")
include("batch.jl")
include("seq.jl")
include("iter.jl")

end
