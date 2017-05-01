export AArray, unsqueeze

const AArray = AbstractArray

initn(dims...) = randn(dims...)/100

unsqueeze(xs, dim = 1) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))
Base.squeeze(xs) = squeeze(xs, 1)

stack(xs, dim = 1) = cat(dim, unsqueeze.(xs, dim)...)
unstack(xs, dim = 1) = [slicedim(xs, dim, i) for i = 1:size(xs, dim)]

mapt(f, x) = f(x)
mapt(f, xs::Tuple) = map(x -> mapt(f, x), xs)

convertel(T::Type, xs::AbstractArray) = convert.(T, xs)
convertel{T}(::Type{T}, xs::AbstractArray{T}) = xs
