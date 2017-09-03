# Arrays

initn(dims...) = randn(dims...)/100

unsqueeze(xs, dim = 1) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))
squeeze(xs, dim = 1) = Base.squeeze(xs, dim)

stack(xs, dim = 1) = cat(dim, unsqueeze.(xs, dim)...)
unstack(xs, dim = 1) = [slicedim(xs, dim, i) for i = 1:size(xs, dim)]

"""
    onehot('b', ['a', 'b', 'c', 'd']) => [false, true, false, false]

    onehot(Float32, 'c', ['a', 'b', 'c', 'd']) => [0., 0., 1., 0.]

Produce a one-hot-encoded version of an item, given a list of possible values
for the item.
"""
onehot(T::Type, label, labels) = T[i == label for i in labels]
onehot(label, labels) = onehot(Int, label, labels)

"""
    onecold([0.0, 1.0, 0.0, ...],
            ['a', 'b', 'c', ...]) => 'b'

The inverse of `onehot`; takes an output prediction vector and a list of
possible values, and produces the appropriate value.
"""
onecold(y::AbstractVector, labels = 1:length(y)) =
  labels[findfirst(y, maximum(y))]

onecold(y::AbstractMatrix, l...) =
  squeeze(mapslices(y -> onecold(y, l...), y, 1), 1)

flatten(xs) = reshape(xs, size(xs, 1), :)

# Other

function accuracy(m, data)
  n = 0
  correct = 0
  for (x, y) in data
    x, y = tobatch.((x, y))
    n += size(x, 1)
    correct += sum(onecold(m(x)) .== onecold(y))
  end
  return correct/n
end

"""
Returns a function that when invoked, will only be triggered at most once
during `timeout` seconds. Normally, the throttled function will run
as much as it can, without ever going more than once per `wait` duration;
but if you'd like to disable the execution on the leading edge, pass
`leading=false`. To enable execution on the trailing edge, ditto.
"""
function throttle(f, timeout; leading=true, trailing=false)
  cooldown = true
  later = nothing

  function throttled(args...; kwargs...)
    yield()

    if cooldown
      if leading
        f(args...; kwargs...)
      else
        later = () -> f(args...; kwargs...)
      end

      cooldown = false
      @schedule try
        while (sleep(timeout); later != nothing)
          later()
          later = nothing
        end
      finally
        cooldown = true
      end
    elseif trailing
      later = () -> f(args...; kwargs...)
    end

    nothing
  end
end
