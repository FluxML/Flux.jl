# Batches

struct Batch{T,S} <: Batchable{T,S}
  data::Storage{T,S}
  Batch{T,S}(data::Storage{T,S}) where {T,S} = new{T,S}(data)
end

Batch(data::Storage{T,S}) where {T,S} = Batch{T,S}(data)

Batch(xs) = Batch(Storage(xs))
Batch{T,S}(xs) where {T,S} = Batch{T,S}(Storage{T,S}(xs))

storage(b::Batch) = b.data

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
  Seq{T,S}(data::Storage{T,S}) where {T,S} = new{T,S}(data)
end

Seq(data::Storage{T,S}) where {T,S} = Seq{T,S}(data)

Seq(xs) = Seq(Storage(xs))
Seq{T,S}(xs) where {T,S} = Seq{T,S}(Storage{T,S}(xs))

storage(s::Seq) = s.data

Base.rpad{T}(xs::Seq{T}, n::Integer, x::T) =
  n-length(xs) ≤ 0 ? xs : vcat(xs, typeof(xs)(repeated(x, n-length(xs))))
