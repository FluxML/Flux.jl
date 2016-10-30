export onehot, onecold, chunk, partition, batches, sequences

onehot(T::Type, label, labels) = T[i == label for i in labels]
onehot(label, labels) = onehot(Int, label, labels)
onecold(pred, labels = 1:length(pred)) = labels[findfirst(pred, maximum(pred))]

using Iterators
import Iterators: partition

export partition

Base.length(l::Iterators.Partition) = length(l.xs) ÷ l.step

_partition(r::UnitRange, step::Integer) = (step*(i-1)+1:step*i for i in 1:(r.stop÷step))
_partition(xs, step) = (xs[i] for i in _partition(1:length(xs), step))

chunk(xs, n) = _partition(xs, length(xs)÷n)

batches(xs...) = (Batch(x) for x in zip(xs...))
sequences(xs, len) = (Seq(x) for x in partition(xs, len))
