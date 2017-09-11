# Arrays

initn(dims...) = randn(dims...)/100

flatten(xs) = reshape(xs, size(xs, 1), :)

unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))

stack(xs, dim) = cat(dim, unsqueeze.(xs, dim)...)
unstack(xs, dim) = [slicedim(xs, dim, i) for i = 1:size(xs, dim)]

# Other

function accuracy(m, data)
  n = 0
  correct = 0
  for (x, y) in data
    x, y = tobatch.((x, y))
    n += size(x, 1)
    correct += sum(argmax(m(x)) .== argmax(y))
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
