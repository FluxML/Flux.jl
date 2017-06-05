module Batches

using Juno, Lazy
using Juno: Tree, Row

export Batch, Batched, Seq, rawbatch, batchone

include("catmat.jl")
include("batch.jl")
include("iter.jl")

end
