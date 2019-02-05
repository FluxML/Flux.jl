using ForwardDiff

seed(x::Real, ::Val) = Dual(x, true)

function seed(x, ::Val{N}, offset = 0) where N
  map(x, reshape(1:length(x), size(x))) do x, i
    Dual(x, ntuple(j -> j+offset == i, Val(N)))
  end
end

extract(x::ForwardDiff.Dual) = x.value, [x.partials...]

function extract(xs::AbstractArray{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
  J = similar(xs, V, N, length(xs))
  for i = 1:length(xs), j = 1:N
    J[j, i] = xs[i].partials.values[j]
  end
  return map(x -> x.value, xs), J
end

function forward_jacobian(f, x, ::Val{N}) where N
  y, _J = extract(f(seed(x, Val(N))))
  J = similar(_J, length(x), length(y))
  J[1:N,:] = _J
  offset = 0
  while offset + N < length(x)
    offset += N
    _, _J = extract(f(seed(x, Val(N), offset)))
    range = (1+offset):min(N+offset,length(x))
    J[range,:] = @view _J[range.-offset,:]
  end
  return y, J
end

function forward_jacobian(f, x)
  if length(x) < ForwardDiff.DEFAULT_CHUNK_THRESHOLD
    forward_jacobian(f, x, Val(length(x)))
  else
    forward_jacobian(f, x, Val(ForwardDiff.DEFAULT_CHUNK_THRESHOLD))
  end
end

forwarddiff(f, x) = istracked(x) ? track(forwarddiff, f, x) : f(x)

vec_scalar(x) = vec(x)
vec_scalar(x::Real) = [x]
reshape_scalar(x, y) = reshape(y, size(x))
reshape_scalar(x::Real, y) = y[]

@grad function forwarddiff(f, x)
  y, J = forward_jacobian(f, data(x))
  return y, ȳ -> (nothing, reshape_scalar(x, J*vec_scalar(ȳ)))
end
