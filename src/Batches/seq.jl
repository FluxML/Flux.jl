struct Seq{T,S} <: ABatch{T}
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

function rebatchseq(xs)
  dims = ndims(xs)-2
  T = Array{eltype(xs),dims}
  S = Array{eltype(xs),dims+1}
  B = Array{eltype(xs),dims+2}
  Batch{Seq{T,S},B}(xs)
end
