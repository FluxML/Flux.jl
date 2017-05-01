export Batch, batchone, tobatch

struct Batch{T,S} <: AbstractVector{T}
  data::CatMat{T,S}
end

@forward Batch.data size, eltype, getindex, setindex!, rawbatch

Batch(xs) = Batch(CatMat(xs))

convert{T,S}(::Type{Batch{T,S}},storage::S) =
  Batch{T,S}(storage)

@render Juno.Inline b::Batch begin
  Tree(Row(Text("Batch of "), eltype(b),
           Juno.fade("[$(length(b))]")),
       Juno.trim(collect(b)))
end

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
