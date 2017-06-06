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
  squeeze(mapslices(y -> onecold(y, l...), y, 2), 2)

chunk(xs, n) = Base.Iterators.partition(xs, length(xs)÷n)
