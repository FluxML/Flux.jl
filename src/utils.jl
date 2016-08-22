export AArray, onehot, onecold

const AArray = AbstractArray

onehot(label, labels) = [i == label for i in labels]
onecold(pred, labels = 1:length(pred)) = labels[findfirst(pred, maximum(pred))]
