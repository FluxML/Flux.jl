export Batch, batchone

struct Batch{T,S} <: AbstractVector{T}
  data::CatMat{T,S}
end

@forward Batch.data size, eltype, getindex, setindex!, rawbatch

Batch(xs) = Batch(CatMat(xs))

convert{T,S}(::Type{Batch{T,S}},storage::S) =
  Batch{T,S}(storage)

@render Juno.Inline b::Batch begin
  Tree(Row(Text("Batch of "), eltype(b),
           Juno.fade("[$(length(b))]")),
       Juno.trim(collect(b)))
end

function rebatch(xs)
  dims = ndims(xs)-1
  T = Array{eltype(xs),dims}
  B = Array{eltype(xs),dims+1}
  Batch{T,B}(xs)
end

convertel(T::Type, xs::Batch) =
  eltype(eltype(xs)) isa T ? xs :
    Batch(map(x->convertel(T, x), xs))

# Add batching semantics to functions operating on raw arrays
# TODO: remove this in favour of full batching semantics

mapt(f, x) = f(x)
mapt(f, xs::Tuple) = map(x -> mapt(f, x), xs)

batchone(x) = Batch((x,))
batchone(x::Batch) = x

function unbatchone(xs::Batch)
  @assert length(xs) == 1
  return first(xs)
end

isbatched(x) = false
isbatched(x::Batch) = true
isbatched(xs::Tuple) = any(isbatched, xs)

batchify(xs) = isbatched(xs) ? (xs, true) : (mapt(batchone, xs), false)

function runbatched(f, xs...)
  # TODO: decide what to do with mixed inputs
  xs, batched = batchify(xs)
  ys = f(xs...)
  batched ? ys : mapt(unbatchone, ys)
end

runrawbatched(f, xs...) =
  runbatched((xs...) -> mapt(rebatch,
                             f(mapt(rawbatch, xs)...)),
             xs...)
