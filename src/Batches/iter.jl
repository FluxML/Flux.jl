import Base: start, next, done, iteratorsize, iteratoreltype, eltype, length

# Stateful iteration

mutable struct StatefulIter{I,S,T}
  iter::I
  state::S
  next::Nullable{T}
end

function StatefulIter(itr)
  state = start(itr)
  val, state = done(itr, state) ? (Nullable(), state) : next(itr, state)
  return StatefulIter(itr, state, convert(Nullable, val))
end

peek(s::StatefulIter) = get(s.next)

function Base.take!(s::StatefulIter)
  x = peek(s)
  if !done(s.iter, s.state)
    s.next, s.state = next(s.iter, s.state)
  else
    s.next = Nullable()
  end
  return x
end

Base.isempty(s::StatefulIter) = isnull(s.next)
Base.eltype(s::StatefulIter) = eltype(s.next)

function taken!(s::StatefulIter, n::Integer)
  xs = eltype(s)[]
  for _ = 1:n
    isempty(s) && break
    push!(xs, take!(s))
  end
  return xs
end

# Batched

struct Batched{I<:StatefulIter,S}
  itr::I
  buf::S
end

function Batched(itr, n::Integer)
  n >= 1 || throw(ArgumentError("batch size must be >= 1"))
  itr = StatefulIter(itr)
  x = peek(itr)
  buf = convert(Batch{typeof(peek(itr))},
                similar(rawbatch(x), n, size(rawbatch(x))...))
  Batched(itr, buf)
end

iteratoreltype(::Type{<:Batched}) = Base.HasEltype()
iteratorsize(::Type{<:Batched}) = Base.SizeUnknown()

eltype{T,S}(x::Batched{T,S}) = S

start(::Batched) = ()

next(x::Batched, _) = x.buf, ()

# will be less hacky if https://github.com/JuliaLang/julia/issues/18823
function done(x::Batched, _)
  next = taken!(x.itr, length(x.buf))
  length(next) < length(x.buf) && return true
  for (i, n) in enumerate(next)
    x.buf[i] = rawbatch(n)
  end
  return false
end
