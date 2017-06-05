import Base: eltype, size, getindex, setindex!, convert, typename

rawbatch(xs) = xs

# Generic methods

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

typerender(B::Type) = B
typerender(B::Type{<:BatchLike}) =
  Row(Juno.typ("$(typename(B).name)"), text"{", typerender(eltype(B)), text"}")

@render Juno.Inline b::BatchLike begin
  Tree(Row(typerender(typeof(b)),
           Juno.fade("[$(length(b))]")),
       Juno.trim(collect(b)))
end

# Concrete storage

struct Storage{T,S} <: BatchLike{T,S}
  data::S
end

allequal(xs) = all(x -> x == first(xs), xs)

function Storage{T,S}(xs, storage::S) where {T, S}
  @assert allequal(map(size, xs))
  @assert size(storage) == (length(xs), size(first(xs))...)
  for i = 1:length(xs)
    storage[i, :] = xs[i]
  end
  return Storage{T,S}(storage)
end

function Storage{T}(xs) where T
  xs′ = map(rawbatch, xs)
  storage = similar(first(xs′), (length(xs′), size(first(xs′))...))
  Storage{T,typeof(storage)}(xs′, storage)
end

Storage(xs) = Storage{eltype(xs)}(xs)

convert{T,S}(B::Type{<:BatchLike{T,S}}, data::S) = B(data)

# Horrible type hacks follow this point

deparam(T::Type) = typename(T).wrapper

dimless(T::Type{<:AbstractArray}) = ndims(T) == 1 ? eltype(T) : deparam(T){eltype(T),ndims(T)-1}

btype(B::Type{<:BatchLike}, S::Type{<:AbstractArray}) = B{dimless(S),S}
btype(B::Type{<:BatchLike{T}} where T, S::Type{<:AbstractArray}) = B{S}
btype(B::Type{<:BatchLike{<:BatchLike}}, S::Type{<:AbstractArray}) =
  deparam(B){btype(eltype(B), dimless(S)),S}

convert{T<:BatchLike}(::Type{T}, xs::AbstractArray) =
  convert(btype(T, typeof(xs)), xs)

convert{T<:BatchLike}(::Type{T}, x::T) = x
