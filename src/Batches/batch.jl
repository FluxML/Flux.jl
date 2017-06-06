# Batches

struct Batch{T,S} <: Batchable{T,S}
  data::Storage{T,S}
end

storage(b::Batch) = b.data

Batch(xs) = Batch(Storage(xs))

convertel(T::Type, xs::Batch) =
  eltype(eltype(xs)) isa T ? xs :
    Batch(map(x->convertel(T, x), xs))

batchone(x) = Batch((x,))
batchone(x::Batch) = x

tobatch(xs::Batch) = rawbatch(xs)
tobatch(xs) = tobatch(batchone(xs))

# Sequences

struct Seq{T,S} <: Batchable{T,S}
  data::Storage{T,S}
end

storage(s::Seq) = s.data

Seq(xs) = Seq(Storage(xs))
