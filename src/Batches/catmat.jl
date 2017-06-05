import Base: eltype, size, getindex, setindex!, convert

ABatch{T} = AbstractVector{T}

struct CatMat{T,S} <: ABatch{T}
  data::S
end

convert{T,S}(::Type{CatMat{T,S}},storage::S) =
  CatMat{T,S}(storage)

eltype{T}(::CatMat{T}) = T

size(b::CatMat) = (size(b.data, 1),)

getindex(b::CatMat, i)::eltype(b) = slicedim(b.data, 1, i)

setindex!(b::CatMat, v, i::Integer) = b.data[i, :] = v

function setindex!(b::CatMat, xs, ::Colon)
  for (i, x) in enumerate(xs)
    b[i] = x
  end
end

allequal(xs) = all(x -> x == first(xs), xs)

function (::Type{CatMat{T,S}}){T,S}(xs, storage::S)
  @assert allequal(map(size, xs))
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
