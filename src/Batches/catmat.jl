import Base: eltype, size, getindex, setindex!, convert

rawbatch(xs) = xs

abstract type BatchLike{T,S} <: AbstractVector{T}
end

eltype{T}(::BatchLike{T}) = T

rawbatch(xs::BatchLike) = rawbatch(xs.data)

size(b::BatchLike) = (size(rawbatch(b), 1),)

getindex(b::BatchLike, i)::eltype(b) = slicedim(rawbatch(b), 1, i)

setindex!(b::BatchLike, v, i::Integer) = rawbatch(b)[i, :] = v

function setindex!(b::BatchLike, xs, ::Colon)
  for (i, x) in enumerate(xs)
    b[i] = x
  end
end

typename(b::Type) = b
typename(b::Type{<:BatchLike}) =
  Row(Juno.typ("$(b.name.name)"), text"{", typename(eltype(b)), text"}")

@render Juno.Inline b::BatchLike begin
  Tree(Row(typename(typeof(b)),
           Juno.fade("[$(length(b))]")),
       Juno.trim(collect(b)))
end

struct Storage{T,S} <: BatchLike{T,S}
  data::S
end

convert{T,S}(::Type{Storage{T,S}},storage::S) =
  Storage{T,S}(storage)

allequal(xs) = all(x -> x == first(xs), xs)

function (::Type{Storage{T,S}}){T,S}(xs, storage::S)
  @assert allequal(map(size, xs))
  @assert size(storage) == (length(xs), size(first(xs))...)
  for i = 1:length(xs)
    storage[i, :] = xs[i]
  end
  return Storage{T,S}(storage)
end

function (::Type{Storage{T}}){T}(xs)
  xs′ = map(rawbatch, xs)
  storage = similar(first(xs′), (length(xs′), size(first(xs′))...))
  Storage{T,typeof(storage)}(xs′, storage)
end

function Storage(xs)
  xs = promote(xs...)
  Storage{eltype(xs)}(xs)
end
