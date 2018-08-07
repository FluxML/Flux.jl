struct IdSet{T} <: AbstractSet{T}
  dict::IdDict{T,Nothing}
  IdSet{T}() where T = new(IdDict{T,Nothing}())
end

Base.eltype(::IdSet{T}) where T = T

IdSet() = IdSet{Any}()

Base.push!(s::IdSet{T}, x::T) where T = (s.dict[x] = nothing; s)
Base.delete!(s::IdSet{T}, x::T) where T = (delete!(s.dict, x); s)
Base.in(x, s::IdSet) = haskey(s.dict, x)

(::Type{IdSet{T}})(xs) where T = push!(IdSet{T}(), xs...)

IdSet(xs) = IdSet{eltype(xs)}(xs)

Base.collect(s::IdSet) = Base.collect(keys(s.dict))
Base.similar(s::IdSet, T::Type) = IdSet{T}()

@forward IdSet.dict Base.length

Base.start(s::IdSet) = start(keys(s.dict))
Base.next(s::IdSet, st) = next(keys(s.dict), st)
Base.done(s::IdSet, st) = done(keys(s.dict), st)
