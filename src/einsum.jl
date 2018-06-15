using MacroTools

_permutedims(x, p) =
  p == collect(1:length(p)) ? x : # TODO 0.7
  p == [2, 1] ? :(transpose($x)) :
  :(permutedims($x, ($(p...),)))

_size(x, n) = Any[:(size($x, $i)) for i = 1:n]

_reshape(x, n, s) = # TODO use transpose
  s == _size(x, n) ? x :
  :(reshape($x, ($(s...),)))

function _expanddims(x, n, ds) # TODO use transpose
  is = _size(x, n)
  foreach(d -> insert!(is, d, 1), ds)
  :(reshape($x, ($(is...),)))
end

function _squeezedims(x, n, ds)
  is = [d for (i, d) in enumerate(_size(x, n)) if i ∉ ds]
  :(reshape($x, ($(is...),)))
end

function _einsum_pair(a, b, dims)
  (a, adims), (b, bdims) = a, b # TODO 0.7
  preserved = setdiff(intersect(adims, bdims), dims)
  broadcast = map(adims -> reduce(setdiff, (adims, preserved, dims)), (adims, bdims)) # TODO 0.7
  # TODO move preserved dims last
  aperm = sortperm(adims, by = i -> i in preserved ? -1 : i in broadcast[1] ? 0 : 1)
  bperm = sortperm(bdims, by = i -> i in preserved ? -1 : i in dims ? 0 : 1)
  a, b = _permutedims(a, aperm), _permutedims(b, bperm)
  adims, bdims = adims[aperm], bdims[bperm]
  if isempty(dims)
    b = _expanddims(b, length(bdims), length(preserved)+(1:length(broadcast[1])))
    :($a .* $b), vcat(adims[aperm], bdims[bperm][length(preserved)+1:end])
  else
    prod(xs) = isempty(xs) ? 0 : length(xs) == 1 ? xs[1] : :(prod(($(xs...),)))

    ashape = _size(a, length(adims))
    npreserve = prod(ashape[1:length(preserved)])
    aaxes = 1+length(preserved):length(adims)-length(dims)
    abroadcast = prod(ashape[aaxes])
    asum = prod(ashape[end-length(dims)+1:end])
    a = _reshape(a, length(adims), [abroadcast, asum]) # TODO preserve

    bshape = _size(b, length(bdims))
    bsum = prod(bshape[length(preserved)+1:end-length(broadcast[2])])
    baxes = length(bdims)-length(broadcast[2])+1:length(bdims)
    bbroadcast = prod(bshape[baxes])
    b = _reshape(b, length(bdims), [bsum, bbroadcast]) # TODO preserve

    ab = :($a*$b)
    shape = vcat(ashape[[1:length(preserved)..., aaxes...]], bshape[baxes])
    shape == [abroadcast, bbroadcast] || (ab = _reshape(ab, 2, shape))
    axes = vcat(adims[[1:length(preserved)..., aaxes...]], bdims[baxes])

    return ab, axes
  end
end

# _einsum_pair([:a, [:i, :j]], [:b, [:j, :k]], [:j])
# _einsum_pair([:a, [:i, :j, :N]], [:b, [:j, :k, :N]], [:j])

macro einsum(ex)
  @capture(ex, [out__] -> *(in__) | in_) || error("`@einsum [...] -> a[...] * b[...] * ...`")
  in isa Vector || (in = [in])
  # TODO rebinding, check dims
  in = map(in) do x
    @capture(x, a_[i__]) || error("Einsum input should be `a[i...]`, got `$x`")
    esc(a), i
  end
  all(length(unique(is)) == length(is) for (_, is) in in) || error("Diagonals not supported")
  labels = unique(vcat(map(x -> x[2], in)...))
  for i in labels
    count(in -> i ∈ in[2], in) > 2 && error("Not supported: index $i appears more than twice")
  end
  y = in[1]
  for i = 1:length(in)-1
    dims = setdiff(union(y[2], in[i+1][2]), out)
    y = _einsum_pair(y, in[i+1], dims)
  end
  reduce = setdiff(y[2], out)
  if !isempty(reduce)
    r = indexin(reduce, y[2])
    y = _squeezedims(:(sum($(y[1]), ($(r...),))), length(y[2]), r),
        setdiff(y[2], reduce)
  end
  @assert sort(y[2]) == sort(out)
  return _permutedims(y[1], indexin(out, y[2]))
end

# @expand @einsum [i] -> a[i,j]
# @expand @einsum [i,k] -> a[i,j] * b[j,k]
# @expand @einsum [i,k] -> a[j,k] * b[i,j]
# @expand @einsum [i,k,N] -> a[i,j,N] * b[j,k,N]
# @expand @einsum [i,j] -> a[i] * b[j]
