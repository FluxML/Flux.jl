# Arrays

initn(dims...) = randn(dims...)/100

flatten(xs) = reshape(xs, size(xs, 1), :)

unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))

stack(xs, dim) = cat(dim, unsqueeze.(xs, dim)...)
unstack(xs, dim) = [slicedim(xs, dim, i) for i = 1:size(xs, dim)]

batchindex(xs, i) = (reverse(Base.tail(reverse(indices(xs))))..., i)

function batch(xs)
  data = similar(first(xs), size(first(xs))..., length(xs))
  for (i, x) in enumerate(xs)
    data[batchindex(data, i)...] = x
  end
  return data
end

Base.rpad(v::AbstractVector, n::Integer, p) = [v; fill(p, max(n - length(v), 0))]

function batchseq(xs, pad, n = maximum(length(x) for x in xs))
  xs_ = [rpad(x, n, pad) for x in xs]
  [batch([xs_[j][i] for j = 1:length(xs_)]) for i = 1:n]
end

# Other

"""
    accuracy(m, x, y, outclass)

classification accuracy

- `m`: model
- `x`: input
- `y`: label
- `outclass`: the output classes of model

# Example
```julia
julia> A
3×5 Array{Float64,2}:
 0.533278  0.176326  0.883837  0.306942  0.744581
 0.193528  0.91559   0.474799  0.437339  0.00711155
 0.973447  0.492269  0.586265  0.371859  0.313075

julia> Flux.accuracy(softmax, A, [:c1, :c2, :c2, :c1, :c3], [:c1, :c2, :c3])
0.2
```
"""
function accuracy(m, x, y, outclass = [])
  pred = isempty(outclass) ?  argmax(m(x)) : argmax(m(x), outclass)
  sum(pred .== y) / length(y)
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
