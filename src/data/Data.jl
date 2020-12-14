module Data

using Random: shuffle!
using Base: @propagate_inbounds

include("dataloader.jl")
export DataLoader


## TODO: remove in v0.13 ##############
module MNIST
images() = error("Dataset is deprecated, use MLDatasets.jl instead.")
labels() = error("Dataset is deprecated, use MLDatasets.jl instead.")
end
module Iris
features() = error("Dataset is deprecated, use MLDatasets.jl instead.")
labels() = error("Dataset is deprecated, use MLDatasets.jl instead.")
end
module FashionMNIST
images() = error("Dataset is deprecated, use MLDatasets.jl instead.")
labels() = error("Dataset is deprecated, use MLDatasets.jl instead.")
end 
module CMUDict
phones() = error("Dataset is deprecated, use MLDatasets.jl instead.")
symbols() = error("Dataset is deprecated, use MLDatasets.jl instead.")
rawdict() = error("Dataset is deprecated, use MLDatasets.jl instead.")
cmudict() = error("Dataset is deprecated, use MLDatasets.jl instead.")
end
module Sentiment
train() = error("Dataset is deprecated, use MLDatasets.jl instead.")
test() = error("Dataset is deprecated, use MLDatasets.jl instead.")
dev() = error("Dataset is deprecated, use MLDatasets.jl instead.")
end

export MNIST, Iris, FashionMNIST, CMUDict, Sentiment
#########################################

end#module