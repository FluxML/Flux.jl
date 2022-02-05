module Data

using Random: shuffle!
using Base: @propagate_inbounds

include("dataloader.jl")
export DataLoader

end#module
