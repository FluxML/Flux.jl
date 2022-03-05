"""
    isloadleaf(x)

Return `true` whenever `x` should be treated as a "leaf node"
for the purposes of loading parameters.
By default, `isloadleaf` returns `true` if [`Functors.isleaf`](@ref)
is `true` for all [`Functors.children(x)`](@ref `Functors.children`).

You can override this function for a specific type if needed.
"""
isloadleaf(x) = all(Functors.isleaf, Functors.children(x))

"""
    loadleaf!(x, x̄, err)

Copy `x̄` to `x` or throw `err` when their sizes are mismatched.
By default, use `copyto!` when `x` and `x̄` are arrays.
Otherwise, just return `x`.
"""
loadleaf!(x, x̄, err) = x
function loadleaf!(x::AbstractArray, x̄, err)
  x .= x̄
  return x
end
function loadleaf!(x::AbstractArray, x̄::AbstractArray, err)
  (size(x) == size(x̄)) || throw(err)
  copyto!(x, x̄)
end

function _loadto!(m, m̄)
  ls, _ = functor(m)
  l̄s, _ = functor(m̄)
  (keys(ls) == keys(l̄s)) ||
    throw(ArgumentError("Tried to load $m̄ into $m but the structures do not match."))

  err = DimensionMismatch("Tried to load $m̄ into $m but the parameter sizes do not match.")
  foreach((l, l̄) -> loadleaf!(l, l̄, err), ls, l̄s)

  return m
end
function loadto!(m::T, m̄::S) where {T, S}
  (nameof(T) == nameof(S)) || throw(ArgumentError("Tried to load $m̄ into $m."))
  _loadto!(m, m̄)
end

"""
    loadmodel!(m, m̄)

Copy all the parameters (trainable and non-trainable) from `m̄` to `m`.

`loadmodel!` recursively walks `m` and `m̄` until it encounters
a subfield, `x`, (i.e. layer) where `isloadleaf(x)` is true.
The parameters of the matching subfield, `x̄`, are copied to `x`,
throwing an error whenever:
- `x` and `x̄` are not the same type (e.g. loading a `Conv` to a `Dense`)
- `x` and `x̄` do not share the same fields
- the parameter sizes are mismatched between `x` and `x̄`

See [`loadleaf!`](@ref) for more details on the copy behavior.
See [`isloadleaf`](@ref) for more details on which layers are considered leaves.

!!! warning
    This function allows `m̄` to be a vector or `Params` for backwards-compatibility.
    You should avoid using `loadmodel!` this way, because it skips most of the structural
    checking used when `m̄` is also a struct. Silent errors may occur.
"""
function loadmodel!(m, xs::Params)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end
loadmodel!(m, xs::AbstractVector) = loadmodel!(m, params(xs))
loadmodel!(m, m̄) = fmap(loadto!, m, m̄; exclude = isloadleaf)
