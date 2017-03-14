export seq, Seq, BatchSeq

struct Seq{T,S} <: AbstractVector{T}
  data::CatMat{T,S}
end

@forward Seq.data size, eltype, getindex, setindex!, rawbatch

Seq(xs) = Seq(CatMat(xs))

convert{T,S}(::Type{Seq{T,S}},storage::S) =
  Seq{T,S}(storage)

@render Juno.Inline b::Seq begin
  Tree(Row(Text("Seq of "), eltype(b),
           Juno.fade("[$(length(b))]")),
       Juno.trim(collect(b)))
end

BatchSeq{T<:Seq} = Batch{T}
