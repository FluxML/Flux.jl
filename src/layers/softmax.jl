mutable struct Softmax{T,N,A,B} <: AbstractArray{T,N}
  logits::A
  probs::B
  Softmax{T,N,A,B}(logits::A) where {T,N,A,B} = new(logits)
end

Softmax(logits::AbstractVecOrMat{<:AbstractFloat}) =
  Softmax{eltype(logits),ndims(logits),typeof(logits),typeof(Tracker.data(logits))}(logits)

@forward Softmax.logits Base.size

Base.IndexStyle(::Type{Softmax{T,N,A}}) where {T,N,A} = IndexStyle(A)

function Base.getindex(s::Softmax, i)
  isdefined(s, :probs) || (s.probs = NNlib.softmax(Tracker.data(s.logits)))
  s.probs[i]
end

softmax(xs::AbstractVecOrMat{<:AbstractFloat}) = Softmax(xs)

softmax(xs::AbstractVecOrMat{<:Real}) = softmax(convert.(AbstractFloat, xs))

softmax(xs::TrackedArray) = TrackedArray(Tracker.Call(NNlib.softmax, xs), Softmax(xs))
