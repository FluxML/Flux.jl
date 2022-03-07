"""
    loadleaf!(dst, src, err)

Copy `src` to `dst` or throw `err` when their sizes are mismatched.
By default, use `copyto!` when `dst` and `src` are arrays.
When only `dst` is an array, set every element to `src`.
Otherwise, just return `dst`.
"""
loadleaf!(dst, src, err) = dst
function loadleaf!(dst::AbstractArray, src, err)
  dst .= src
  return dst
end
function loadleaf!(dst::AbstractArray, src::AbstractArray, err)
  (size(dst) == size(src)) || throw(err)
  copyto!(dst, src)
end

"""
    loadmodel!(dst, src)

Copy all the parameters (trainable and non-trainable) from `src` to `dst`.

`loadmodel!` recursively walks the [`Functors.children`](@ref) of `dst` and `src`
calling `loadleaf!` on any pair of children where [`Functors.isleaf`](@ref) is true.
It throws an error whenever:
- `dst` and `src` do not share the same fields (at any level)
- the sizes of leaf nodes are mismatched between `dst` and `src`

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

julia> dst = loadmodel!(dst, src)
Chain(
  Dense(5 => 2),                        # 12 parameters
  Dense(2 => 1),                        # 3 parameters
)                   # Total: 4 arrays, 15 parameters, 316 bytes.

julia> all(isone, dst[1].weight)
false

julia> dst[1].weight == src[1].weight
true

julia> dst[2].bias == src[2].bias
true
```

See [`Flux.loadleaf!`](@ref) for more details on the copy behavior.

!!! warning
    This function allows `src` to be a `Params` for backwards-compatibility.
    You should avoid using `loadmodel!` this way, because it skips most of the structural
    checking used when `src` is also a nested structure. Silent errors may occur.
"""
function loadmodel!(m, xs::Params)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end
function loadmodel!(dst, src)
  ldsts, _ = functor(dst)
  lsrcs, _ = functor(src)
  (keys(ldsts) == keys(lsrcs)) ||
    throw(ArgumentError("Tried to load $src into $dst but the structures do not match."))

  err = DimensionMismatch("Tried to load $src into $dst but the parameter sizes do not match.")
  foreach(ldsts, lsrcs) do ldst, lsrc
    Functors.isleaf(ldst) ? loadleaf!(ldst, lsrc, err) : loadmodel!(ldst, lsrc)
  end

  return dst
end
