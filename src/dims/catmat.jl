import Base: eltype, size, getindex, setindex!, convert

export CatMat, rawbatch

struct CatMat{T,S} <: AbstractVector{T}
  data::S
end

convert{T,S}(::Type{CatMat{T,S}},storage::S) =
  CatMat{T,S}(storage)

eltype{T}(::CatMat{T}) = T

size(b::CatMat) = (size(b.data, 1),)

getindex(b::CatMat, i)::eltype(b) = slicedim(b.data, 1, i)

setindex!(b::CatMat, v, i) = b.data[i, :] = v

allequal(xs) = all(x -> x == first(xs), xs)

function (::Type{CatMat{T,S}}){T,S}(xs, storage::S)
  @assert @>> xs map(size) allequal
  @assert size(storage) == (length(xs), size(first(xs))...)
  for i = 1:length(xs)
    storage[i, :] = xs[i]
  end
  return CatMat{T,S}(storage)
end

function (::Type{CatMat{T}}){T}(xs)
  xs′ = map(rawbatch, xs)
  storage = similar(first(xs′), (length(xs′), size(first(xs′))...))
  CatMat{T,typeof(storage)}(xs′, storage)
end

function CatMat(xs)
  xs = promote(xs...)
  CatMat{eltype(xs)}(xs)
end

@render Juno.Inline b::CatMat begin
  Tree(Row(Text("CatMat of "), eltype(b),
           Juno.fade("[$(length(b))]")),
       Juno.trim(collect(b)))
end

rawbatch(xs) = xs

rawbatch(xs::CatMat) = xs.data
