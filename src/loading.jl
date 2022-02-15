isloadleaf(x) = all(Functors.isleaf, Functors.children(x))

loadnumeric!(x, x̄, err) = x
loadnumeric!(x::Zeros, x̄, err) = x
function loadnumeric!(x::AbstractArray, x̄::AbstractArray, err)
    (size(x) == size(x̄)) || throw(err)
    copyto!(x, x̄)
end

function _loadto!(m, m̄)
    ls, _ = functor(m)
    l̄s, _ = functor(m̄)
    (keys(ls) == keys(l̄s)) ||
        throw(ArgumentError("Tried to load $m̄ into $m but the structures do not match."))

    err = DimensionMismatch("Tried to load $m̄ into $m but the parameter sizes do not match.")
    foreach((l, l̄) -> loadnumeric!(l, l̄, err), ls, l̄s)

    return m
end
function loadto!(m::T, m̄::S) where {T, S}
    (nameof(T) == nameof(S)) || throw(ArgumentError("Tried to load $m̄ into $m."))
    _loadto!(m, m̄)
end

function loadmodel!(m, xs::Params)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end
loadmodel!(m, xs::AbstractVector) = loadmodel!(m, params(xs))
loadmodel!(m, m̄) = fmap(loadto!, m, m̄; exclude = isloadleaf)
