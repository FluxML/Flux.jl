call(f, xs...) = f(xs...)

# Arrays

initn(dims...) = randn(dims...)/100

unsqueeze(xs, dim = 1) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))
squeeze(xs, dim = 1) = Base.squeeze(xs, dim)

stack(xs, dim = 1) = cat(dim, unsqueeze.(xs, dim)...)
unstack(xs, dim = 1) = [slicedim(xs, dim, i) for i = 1:size(xs, dim)]

convertel(T::Type, xs::AbstractArray) = convert.(T, xs)
convertel{T}(::Type{T}, xs::AbstractArray{T}) = xs

a ∘ b = a .* b

broadcastto(xs::AbstractArray, shape) = xs .* ones(shape)

# Tuples

mapt(f, x) = f(x)
mapt(f, xs::Tuple) = map(x -> mapt(f, x), xs)

function collectt(xs)
  ys = []
  mapt(x -> push!(ys, x), xs)
  return ys
end

function shapecheckt(xs::Tuple, ys::Tuple)
  length(xs) == length(ys) || error("Expected tuple length $(length(xs)), got $ys")
  shapecheckt.(xs, ys)
end

shapecheckt(xs::Tuple, ys) = error("Expected tuple, got $ys")
shapecheckt(xs, ys) = nothing

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
