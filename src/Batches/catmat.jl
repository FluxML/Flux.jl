import Base: eltype, size, getindex, setindex!, convert

abstract type BatchLike{T,S} <: AbstractVector{T}
end

struct Storage{T,S} <: BatchLike{T,S}
  data::S
end

convert{T,S}(::Type{Storage{T,S}},storage::S) =
  Storage{T,S}(storage)

eltype{T}(::Storage{T}) = T

size(b::Storage) = (size(b.data, 1),)

getindex(b::Storage, i)::eltype(b) = slicedim(b.data, 1, i)

setindex!(b::Storage, v, i::Integer) = b.data[i, :] = v

function setindex!(b::Storage, xs, ::Colon)
  for (i, x) in enumerate(xs)
    b[i] = x
  end
end

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

@render Juno.Inline b::Storage begin
  Tree(Row(Text("Storage of "), eltype(b),
           Juno.fade("[$(length(b))]")),
       Juno.trim(collect(b)))
end

rawbatch(xs) = xs

rawbatch(xs::Storage) = xs.data
