loadleaf!(dst, src, err) = dst
function loadleaf!(dst::AbstractArray, src::Bool, err)
  if iszero(src)
    dst .= src
  else
    error("Cannot copy boolean parameter == true to non-zero parameter.")
  end
  return dst
end
loadleaf!(dst::Bool, src::AbstractArray, err) = iszero(dst) ? dst :
  error("Cannot copy non-zero parameter to boolean parameter == true.")
function loadleaf!(dst::AbstractArray, src::AbstractArray, err)
  (size(dst) == size(src)) || throw(err)
  copyto!(dst, src)
end

_tie_check(dst::Bool, src::AbstractArray) = iszero(dst) ||
  error("Encountered tied parameter with boolean source at some nodes and non-boolean sources at others.")
_tie_check(dst::AbstractArray, src::Bool) = (iszero(dst) && iszero(src)) ||
  error("Encountered tied parameter with boolean source at some nodes and non-boolean sources at others.")
_tie_check(dst::AbstractArray, src::AbstractArray) = (dst == src) ||
  error("Encountered tied destination parameters with untied and mismatched sources.")
_tie_check(dst, src) = true

_bool_tie_check(dst, src) = true

"""
    loadmodel!(dst, src)

Copy all the parameters (trainable and non-trainable) from `src` into `dst`.

Recursively walks `dst` and `src` together using [`Functors.children`](@ref),
and calling `copyto!` on parameter arrays.
Non-array elements (such as activation functions) are not copied
and do not need to match between `dst` and `src`.
Inactive parameters can be encoded by using the boolean value `false` instead of an array.
If `dst == false` and `src` is an all-zero array, no error will be raised (and no values copied);
however, attempting to copy a non-zero array to an inactive parameter will throw an error.
Likewise, copying `src == false` to any `dst` array is valid, but copying `src == true` will error.

Throws an error when:
- `dst` and `src` do not share the same fields (at any level)
- the sizes of leaf nodes are mismatched between `dst` and `src`
- `dst` is a "tied" parameter (i.e. refers to another parameter) and
  loaded into multiple times with mismatched source values

```julia
julia> using Flux: loadmodel!

julia> dst = Chain(Dense(Flux.ones32(2, 5)), Dense(2 => 1))
Chain(
  Dense(5 => 2),                        # 12 parameters
  Dense(2 => 1),                        # 3 parameters
)                   # Total: 4 arrays, 15 parameters, 316 bytes.

julia> src = Chain(Dense(5 => 2), Dense(2 => 1));

julia> all(isone, dst[1].weight)
true

julia> loadmodel!(dst, src);

julia> dst[1].weight ≈ ones(2, 5)
false

julia> dst[1].weight == src[1].weight
true

julia> dst[2].bias == src[2].bias
true
```
"""
function loadmodel!(dst, src; cache = Base.IdSet())
  ldsts, _ = functor(dst)
  lsrcs, _ = functor(src)
  (keys(ldsts) == keys(lsrcs)) ||
    throw(ArgumentError("Tried to load $src into $dst but the structures do not match."))

  err = DimensionMismatch("Tried to load $src into $dst but the parameter sizes do not match.")
  foreach(ldsts, lsrcs) do ldst, lsrc
    if ldst in cache # we already loaded this parameter before
      _tie_check(ldst, lsrc) && return ldst
    elseif Functors.isleaf(ldst) # our first time loading this leaf
      push!(cache, ldst)
      loadleaf!(ldst, lsrc, err)
    else # this isn't a leaf
      loadmodel!(ldst, lsrc; cache = cache)
    end
  end

  return dst
end
