export Seq, BatchSeq

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

function rebatchseq(xs)
  dims = ndims(xs)-2
  T = Array{eltype(xs),dims}
  S = Array{eltype(xs),dims+1}
  B = Array{eltype(xs),dims+2}
  Batch{Seq{T,S},B}(xs)
end

# SeqModel wrapper layer for convenience

struct SeqModel <: Model
  model
  steps::Int
end

runseq(f, xs::Tuple...) = f(xs...)
runseq(f, xs::AbstractArray...) = stack(f(map(x -> (unstack(x,2)...,), xs)...), 2)
runseq(f, xs::BatchSeq...) = rebatchseq(runseq(f, rawbatch.(xs)...))

function (m::SeqModel)(x)
  runseq(x) do x
    @assert length(x) == m.steps "Expected seq length $(m.steps), got $(size(x, 2))"
    m.model(x)
  end
end

back!(m::SeqModel, Δ, x) = (runseq((Δ, x) -> back!(m.model, Δ, x)[1], Δ, x),)

update!(m::SeqModel, η) = update!(m.model, η)
