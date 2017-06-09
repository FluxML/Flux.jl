import Base: eltype, size, getindex, setindex!, convert, typename

# Concrete storage

struct Storage{T,S}
  data::S
  Storage{T,S}(data::S) where {T,S} = new{T,S}(data)
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

function Storage{T,S}(xs) where {T,S}
  xs′ = map(rawbatch, xs)
  storage = S(length(xs′), size(first(xs′))...)
  Storage{T,typeof(storage)}(xs′, storage)
end

Storage{T}(xs) where T = Storage{T,diminc(typeof(rawbatch(first(xs))))}(xs)

Storage(xs) = Storage{eltype(xs)}(xs)

convert{T,S}(::Type{Storage{T,S}}, data::S) = Storage{T,S}(data)

convert{T}(::Type{Storage{T}}, data::AbstractArray) = convert(Storage{T,typeof(data)}, data)

# Storage utility methods

rawbatch(xs) = xs
rawbatch(xs::Storage) = xs.data

eltype{T}(::Storage{T}) = T

size(b::Storage) = (size(b.data, 1),)

getindex(b::Storage, i)::eltype(b) = slicedim(b.data, 1, i)

setindex!(b::Storage, v, i::Integer) = b.data[i, :] = v

function setindex!(b::Storage, xs, ::Colon)
  for (i, x) in enumerate(xs)
    b[i] = x
  end
end

# Generic methods

abstract type Batchable{T,S} <: AbstractVector{T}
end

rawbatch(xs::Batchable) = rawbatch(storage(xs))
size(xs::Batchable) = size(storage(xs))
getindex(xs::Batchable, i) = getindex(storage(xs), i)
setindex!(xs::Batchable, v, i...) = setindex!(storage(xs), v, i...)

Base.vcat{T<:Batchable}(xs::T, ys::T)::T = vcat(rawbatch(xs), rawbatch(ys))

typerender(B::Type) = B
typerender(B::Type{<:Batchable}) =
  Row(Juno.typ("$(typename(B).name)"), text"{", typerender(eltype(B)), text"}")

@render Juno.Inline b::Batchable begin
  Tree(Row(typerender(typeof(b)),
           Juno.fade("[$(length(b))]")),
       Juno.trim(collect(b)))
end

# Horrible type hacks follow this point

deparam(T::Type) = typename(T).wrapper

diminc(T::Type) = Vector{T}
diminc(T::Type{<:AbstractArray}) = deparam(T){eltype(T),ndims(T)+1}

dimdec{T}(::Type{<:AbstractArray{T,1}}) = T
dimdec(T::Type{<:AbstractArray}) = deparam(T){eltype(T),ndims(T)-1}

btype(B::Type, S::Type{<:AbstractArray}) = B
btype(B::Type{<:Batchable}, S::Type{<:AbstractArray}) = B{dimdec(S),S}
btype{T}(B::Type{<:Batchable{T}}, S::Type{<:AbstractArray}) = B{S}
btype{T,S<:AbstractArray}(B::Type{<:Batchable{T,S}}, ::Type{S}) = B
btype(B::Type{<:Batchable{<:Batchable}}, S::Type{<:AbstractArray}) =
  deparam(B){btype(eltype(B), dimdec(S)),S}

convert{T<:Batchable}(::Type{Storage{T}}, data::AbstractArray) =
  Storage{btype(T,dimdec(typeof(data))),typeof(data)}(data)

convert{T,S<:AbstractArray}(B::Type{<:Batchable{T,S}}, data::S) = B(convert(Storage{T,S}, data))

convert{B<:Batchable}(::Type{B}, data::AbstractArray) = convert(btype(B,typeof(data)), data)

convert{B<:Batchable}(::Type{B}, xs::B) = xs
