export Batch, batchone

immutable Batch{T,S} <: AbstractVector{T}
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

# Convenience methods for batch size 1

batchone(x) = Batch((x,))
batchone(x::Batch) = x
batchone(x::Tuple) = map(batchone, x)

function unbatchone(xs::Batch)
  @assert length(xs) == 1
  return first(xs)
end

unbatchone(xs::Tuple) = map(unbatchone, xs)

function rebatch(xs)
  dims = ndims(xs)-1
  T = Array{eltype(xs),dims}
  B = Array{eltype(xs),dims+1}
  Batch{T,B}(xs)
end

rebatch(xs::Tuple) = map(rebatch, xs)

convertel(T::Type, xs::Batch) =
  isa(eltype(eltype(xs)), T) ? xs :
    Batch(map(x->convertel(T, x), xs))
