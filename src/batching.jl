export batch

# Treat the first dimension as the batch index
# TODO: custom data type for this
batch(x) = reshape(x, (1,size(x)...))
batch(xs...) = vcat(map(batch, xs)...)

unbatch(xs) = reshape(xs, size(xs)[2:end])
