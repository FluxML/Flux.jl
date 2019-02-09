module FashionMNIST

using ..MNIST: gzopen, imageheader, rawimage, labelheader, rawlabel
using ..Data: download_and_verify

const dir = joinpath(@__DIR__, "../../deps/fashion-mnist")

function load()
  mkpath(dir)
  cd(dir) do
    for (file, hash) in [("train-images-idx3-ubyte", "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"),
                         ("train-labels-idx1-ubyte", "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845"),
                         ("t10k-images-idx3-ubyte" , "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073"),
                         ("t10k-labels-idx1-ubyte" , "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5")]
      isfile(file) && continue
      @info "Downloading Fashion-MNIST dataset"
      download_and_verify("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/$file.gz", "$file.gz", hash)
      open(file, "w") do io
        write(io, gzopen(read, "$file.gz"))
      end
    end
  end
end

const TRAINIMAGES = joinpath(dir, "train-images-idx3-ubyte")
const TRAINLABELS = joinpath(dir, "train-labels-idx1-ubyte")
const TESTIMAGES = joinpath(dir, "t10k-images-idx3-ubyte")
const TESTLABELS = joinpath(dir, "t10k-labels-idx1-ubyte")

"""
    images()
    images(:test)

Load the Fashion-MNIST images.

Each image is a 28×28 array of `Gray` colour values (see Colors.jl).

Returns the 60,000 training images by default; pass `:test` to retreive the
10,000 test images.
"""
function images(set = :train)
  load()
  io = IOBuffer(read(set == :train ? TRAINIMAGES : TESTIMAGES))
  _, N, nrows, ncols = imageheader(io)
  [rawimage(io) for _ in 1:N]
end

"""
    labels()
    labels(:test)

Load the labels corresponding to each of the images returned from `images()`.
Each label is a number from 0-9.

Returns the 60,000 training labels by default; pass `:test` to retreive the
10,000 test labels.
"""
function labels(set = :train)
  load()
  io = IOBuffer(read(set == :train ? TRAINLABELS : TESTLABELS))
  _, N = labelheader(io)
  [rawlabel(io) for _ = 1:N]
end

end
