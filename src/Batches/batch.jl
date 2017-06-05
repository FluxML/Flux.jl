# Batches

struct Batch{T,S} <: BatchLike{T,S}
  data::Storage{T,S}
end

@forward Batch.data size, eltype, getindex, setindex!, rawbatch

Batch(xs) = Batch(Storage(xs))

convert{T,S}(::Type{Batch{T,S}},storage::S) =
  Batch{T,S}(storage)

# TODO: figure out how to express this as a generic convert
function rebatch(xs)
  dims = ndims(xs)-1
  T = Array{eltype(xs),dims}
  B = Array{eltype(xs),dims+1}
  Batch{T,B}(xs)
end

convertel(T::Type, xs::Batch) =
  eltype(eltype(xs)) isa T ? xs :
    Batch(map(x->convertel(T, x), xs))

batchone(x) = Batch((x,))
batchone(x::Batch) = x

tobatch(xs::Batch) = rawbatch(xs)
tobatch(xs) = tobatch(batchone(xs))

# Sequences

struct Seq{T,S} <: BatchLike{T,S}
  data::Storage{T,S}
end

@forward Seq.data size, eltype, getindex, setindex!, rawbatch

Seq(xs) = Seq(Storage(xs))

convert{T,S}(::Type{Seq{T,S}},storage::S) =
  Seq{T,S}(storage)

BatchSeq{T<:Seq} = Batch{T}

function rebatchseq(xs)
  dims = ndims(xs)-2
  T = Array{eltype(xs),dims}
  S = Array{eltype(xs),dims+1}
  B = Array{eltype(xs),dims+2}
  Batch{Seq{T,S},B}(xs)
end
