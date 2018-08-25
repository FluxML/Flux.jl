module MNIST

using GZip, Colors

const Gray = Colors.Gray{Colors.N0f8}

const dir = joinpath(@__DIR__, "../../deps/mnist")

const fileinfo = [
    (name = "train-images-idx3-ubyte", size = 47040016),
    (name = "train-labels-idx1-ubyte", size = 60008),
    (name = "t10k-images-idx3-ubyte", size = 7840016),
    (name = "t10k-labels-idx1-ubyte", size = 10008)]
const TRAINIMAGES, TRAINLABELS, TESTIMAGES, TESTLABELS =
    [joinpath(dir, file.name) for file in fileinfo]

function load()
  mkpath(dir)
  cd(dir) do
    for file in fileinfo
      filesize(file.name) == file.size && continue
      @info "Downloading MNIST dataset: $(file.name)"
      download("https://cache.julialang.org/http://yann.lecun.com/exdb/mnist/$(file.name).gz", "$(file.name).gz")
      open(file.name, "w") do io
        write(io, GZip.open(read, "$(file.name).gz"))
      end
    end
  end
end

const IMAGEOFFSET = 16
const LABELOFFSET = 8

const NROWS = 28
const NCOLS = 28


function imageheader(io::IO)
  magic_number = bswap(read(io, UInt32))
  total_items = bswap(read(io, UInt32))
  nrows = bswap(read(io, UInt32))
  ncols = bswap(read(io, UInt32))
  return magic_number, Int(total_items), Int(nrows), Int(ncols)
end

function labelheader(io::IO)
  magic_number = bswap(read(io, UInt32))
  total_items = bswap(read(io, UInt32))
  return magic_number, Int(total_items)
end

function rawimage(io::IO)
  img = Array{Gray}(undef, NCOLS, NROWS)
  for i in 1:NCOLS, j in 1:NROWS
    img[i, j] = reinterpret(Colors.N0f8, read(io, UInt8))
  end
  return img
end

function rawimage(io::IO, index::Integer)
  seek(io, IMAGEOFFSET + NROWS * NCOLS * (index - 1))
  return rawimage(io)
end

rawlabel(io::IO) = Int(read(io, UInt8))

function rawlabel(io::IO, index::Integer)
  seek(io, LABELOFFSET + (index - 1))
  return rawlabel(io)
end

getfeatures(io::IO, index::Integer) = vec(getimage(io, index))

"""
    images()
    images(:test)

Load the MNIST images.

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

end # module
