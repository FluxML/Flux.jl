module Data

import ..Flux
import SHA

using Random: shuffle!
using Base: @propagate_inbounds

export CMUDict, cmudict

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

include("dataloader.jl")
export DataLoader

include("mnist.jl")
export MNIST

include("fashion-mnist.jl")
export FashionMNIST

include("cmudict.jl")
using .CMUDict

include("tree.jl")
include("sentiment.jl")
using .Sentiment

include("iris.jl")
export Iris

include("housing.jl")
export Housing

@deprecate DataLoader(x...; kws...) DataLoader(x; kws...)

end
