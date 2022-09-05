loadleaf!(dst, src, err) = dst
loadleaf!(dst::AbstractArray, src, err) =
  error("Tried to copy $src into an array destination; this is not allowed.")
loadleaf!(dst, src::AbstractArray, err) =
  error("Tried to copy an array to $dst; this is not allowed.")
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

_filter_children(f, children::NamedTuple) =
  NamedTuple(filter(kv -> f(kv[2]), pairs(children)))
_filter_children(f, children) = filter(f, children)

"""
    loadmodel!(dst, src)

Copy all the parameters (trainable and non-trainable) from `src` into `dst`.

Recursively walks `dst` and `src` together using [`Functors.children`](@ref),
and calling `copyto!` on parameter arrays or throwing an error when there is a mismatch.
Non-array elements (such as activation functions) are not copied and need not match.
Zero bias vectors and `bias=false` are considered equivalent
(see extended help for more details).

# Examples
```julia
julia> dst = Chain(Dense(Flux.ones32(2, 5), Flux.ones32(2), tanh), Dense(2 => 1; bias = [1f0]))
Chain(
  Dense(5 => 2, tanh),                  # 12 parameters
  Dense(2 => 1),                        # 3 parameters
)                   # Total: 4 arrays, 15 parameters, 316 bytes.

julia> dst[1].weight ≈ ones(2, 5)  # by construction
true

julia> src = Chain(Dense(5 => 2, relu), Dense(2 => 1, bias=false));

julia> Flux.loadmodel!(dst, src);

julia> dst[1].weight ≈ ones(2, 5)  # values changed
false

julia> iszero(dst[2].bias)
true
```

# Extended help

Throws an error when:
- `dst` and `src` do not share the same fields (at any level)
- the sizes of leaf nodes are mismatched between `dst` and `src`
- copying non-array values to/from an array parameter
  (except inactive parameters described below)
- `dst` is a "tied" parameter (i.e. refers to another parameter) and
  loaded into multiple times with mismatched source values

Inactive parameters can be encoded by using the boolean value `false` instead of an array.
If `dst == false` and `src` is an all-zero array, no error will be raised (and no values copied);
however, attempting to copy a non-zero array to an inactive parameter will throw an error.
Likewise, copying a `src` value of `false` to any `dst` array is valid,
but copying a `src` value of `true` will error.
"""
function loadmodel!(dst, src; filter = _ -> true, cache = Base.IdSet())
  ldsts = _filter_children(filter, functor(dst)[1])
  lsrcs = _filter_children(filter, functor(src)[1])
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
      loadmodel!(ldst, lsrc; filter = filter, cache = cache)
    end
  end

  return dst
end
