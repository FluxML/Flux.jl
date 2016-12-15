export Batch, batchone

immutable Batch{T,S} <: AbstractVector{T}
  data::CatMat{T,S}
end

@forward Batch.data size, eltype, getindex, setindex!, rawbatch

Batch(xs) = Batch(CatMat(xs))

convert{T,S}(::Type{Batch{T,S}},storage::S) =
  Batch{T,S}(storage)

batchone(x) = Batch((x,))
batchone(x::Batch) = x

Media.render{T<:Batch}(i::Juno.Inline, b::Type{T}) =
  render(i, Row(Juno.typ("Batch"), text"{", eltype(T), text"}"))

@render Juno.Inline b::Batch begin
  Tree(Row(Text("Batch of "), eltype(b),
           Juno.fade("[$(length(b))]")),
       Juno.trim(collect(b)))
end
