unsqueeze(xs, dim = 1) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))
Base.squeeze(xs) = squeeze(xs, 1)

stack(xs, dim = 1) = cat(dim, unsqueeze.(xs, dim)...)
unstack(xs, dim = 1) = [slicedim(xs, dim, i) for i = 1:size(xs, dim)]
