module Data

import ..Flux
import SHA

export CMUDict, cmudict

deps(path...) = joinpath(@__DIR__, "..", "..", "deps", path...)

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
using .CMUDict

include("tree.jl")
include("sentiment.jl")
using .Sentiment

include("iris.jl")
export Iris

end
