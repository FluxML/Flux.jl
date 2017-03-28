unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))

stack(xs, dim = 1) = cat(dim, unsqueeze.(xs, dim)...)
unstack(xs, dim = 1) = [slicedim(xs, dim, i) for i = 1:size(xs, dim)]
