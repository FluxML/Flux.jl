export Batched

zipt(xs...) = (xs,)
zipt(xs::Tuple...) = zip(xs...)

import Base: start, next, done, iteratorsize, iteratoreltype, eltype, length

mutable struct Batched{T,S}
  batch::Int
  iter::T
  "`Batched` always read a batch in advance, and store it in `buf`"
  buf::S
  i
end

function Batched(iter::T, batch::Integer) where T
  batch >= 1 || throw(ArgumentError("batch size must >= 1"))
  i = start(iter)
  done(iter, i) && return Batched{T,Void}(batch, iter, nothing, i)
  v, i = next(iter, i)

  buf = mapt(v) do x
    storage = Array{eltype(x)}(batch, size(x)...)
    storage[1, :] = x
    rebatch(storage)
  end

  for ibatch in 2:batch
    if done(iter, i)
      warn("data less than one batch will be ignored, please use a smaller batch size")
      return Batched{T,Void}(batch, iter, nothing, i)
    end

    v, i = next(iter, i)
    map(x->setindex!(x..., ibatch), zipt(buf, v))
  end

  Batched{T,typeof(buf)}(batch, iter, buf, i)
end

iteratoreltype(::Type{Batched{T,S}}) where {T,S} = Base.HasEltype()

iteratorsize(::Type{Batched{T,S}}) where {T,S} =
  iteratorsize(T) isa Base.HasShape ?
    Base.HasLength() : iteratorsize(T)

length(x::Batched) = length(x.iter) ÷ x.batch

eltype(x::Batched{T,S}) where {T,S} = S

start(x::Batched) = true

next(x::Batched, ::Bool) = x.buf, false

# will be less hacky if https://github.com/JuliaLang/julia/issues/18823
function done(x::Batched, fresh)
  fresh && return false

  for ibatch in 1:x.batch
    if done(x.iter, x.i)
      ibatch != 1 && warn("cannot perfectly divide data by batch size, remainder will be discarded")
      return true
    end

    v, x.i = next(x.iter, x.i)
    map(x->setindex!(x..., ibatch), zipt(x.buf, v))
  end

  false
end

done(::Batched{T,Void}, ::Bool) where T = true
