module Data

using Random: shuffle!
using Base: @propagate_inbounds
using LearnBase
using LearnBase: nobs, getobs

include("dataloader.jl")
export DataLoader

## TODO for v0.13: remove everything below ##############
## Also remove the following deps:
## AbstractTrees, ZipFiles, CodecZLib

import ..Flux
import SHA

deprecation_message() = @warn("Flux's datasets are deprecated, please use the package MLDatasets.jl")

function deps(path...)
  if isnothing(@__DIR__) # sysimages
    joinpath("deps", path...)
  else
    joinpath(@__DIR__, "..", "..", "deps", path...)
  end
end

function download_and_verify(url, path, hash)
    tmppath = tempname()
    download(url, tmppath)
    hash_download = open(tmppath) do f
        bytes2hex(SHA.sha256(f))
    end
    if hash_download !== hash
        msg  = "Hash Mismatch!\n"
        msg *= "  Expected sha256:   $hash\n"
        msg *= "  Calculated sha256: $hash_download"
        error(msg)
    end
    mv(tmppath, path; force=true)
end

function __init__()
  mkpath(deps())
end

include("mnist.jl")
export MNIST

include("fashion-mnist.jl")
export FashionMNIST

include("cmudict.jl")
export CMUDict
using .CMUDict; export cmudict

include("tree.jl")
include("sentiment.jl")
export Sentiment

include("iris.jl")
export Iris

include("housing.jl")
export Housing

#########################################

end#module
