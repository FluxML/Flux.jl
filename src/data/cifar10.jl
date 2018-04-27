module CIFAR10

using BinDeps

const dir = joinpath(@__DIR__, "../../deps/cifar10")
const dir_files = joinpath(dir,"cifar-10-batches-bin")

const NROWS = 32
const NCOLS = 32
const NCHAN = 3
const NTRAINBATCHES = 5
const IMPERBATCH = 10000

const IMAGEPIX = NROWS * NCOLS * NCHAN

const TESTBATCH = joinpath(dir_files,"test_batch.bin")
const LABELS = joinpath(dir_files,"batches.meta.txt")

function load()
  mkpath(dir)
  cd(dir) do
    if(isdir("cifar-10-batches-bin"))
      return
    elseif(isfile("cifar-10-binary.tar.gz"))
      run(unpack_cmd("cifar-10-binary.tar.gz","./",".gz",".tar"))
      rm("cifar-10-binary.tar.gz")
    else
      download("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", "cifar-10-binary.tar.gz")
      run(unpack_cmd("cifar-10-binary.tar.gz","./",".gz",".tar"))
      rm("cifar-10-binary.tar.gz")
    end
  end
end

function rawimage(io::IO)
  img = Array{Float64}(NCHAN,NCOLS,NROWS)
  skip(io,1)
  for i in 1:NCHAN, j in 1:NCOLS, k in 1:NROWS
    img[i, j, k] = float(read(io, UInt8))
  end
  img
end

rawlabel(io::IO) = Int(read(io,UInt8))

function rawimage(io::IO, index::Integer)
  seek(io, IMAGEPIX * (index - 1))
  rawimage(io)
end

function rawlabel(io::IO, index::Integer)
  seek(io, IMAGEPIX * (index - 1))
  rawlabel(io)
end

function imagebatch(batchnum::Integer)
  batch = joinpath(dir_files,"data_batch_$batchnum.bin")
  io = open(batch,"r")
  [rawimage(io) for _ in 1:IMPERBATCH]
end

function labelbatch(batchnum::Integer)
  batch = joinpath(dir_files,"data_batch_$batchnum.bin")
  io = open(batch,"r")
  trainlabel = []
  while(!eof(io))
    push!(trainlabel,rawlabel(io))
    skip(io,IMAGEPIX)
  end
  trainlabel
end

"""
    images()
    images(:test)

Load the CIFAR-10 images.

Each image is a 3×32×32 array of Float64 type where the dimensions are Channel×Column×Row.

Returns the 50,000 training images by default; pass `:test` to retreive the
10,000 test images.
"""
function images(set = :train)
  load()
  if(set == :train)
    trainimgs = []
    for i in 1:NTRAINBATCHES
      trainimgs = vcat(trainimgs, imagebatch(i))
    end
    return trainimgs
  else
    io = open(TESTBATCH, "r")
    return [rawimage(io) for _ in 1:IMPERBATCH]
  end
end

"""
    labels()
    labels(:test)

Load the labels corresponding to each of the images returned from `images()`.
Each label is a number from 0-9.

Returns the 50,000 training labels by default; pass `:test` to retreive the
10,000 test labels.
"""

function labels(set = :train)
  load()
  if(set == :train)
    trainlabels = []
    for i in 1:NTRAINBATCHES
      trainlabels = vcat(trainlabels, labelbatch(i))
    end
    return trainlabels
  else
    io = open(TESTBATCH, "r")
    testlabels = []
    while(!eof(io))
      push!(testlabels,rawlabel(io))
      skip(io,IMAGEPIX)
    end
    return testlabels
  end
end

"""
    labelnames()

Load the category name corresponding to the label number.

Returns a Dictionary where every label number has been mapped to the category title.
"""

function labelnames()
  labelmap = Dict{Int,String}()
  label = 0
  str = ""
  cd(dir_files) do
    file = Char.(read(LABELS))
    for i in file
      if(i=='\n')
        str!="" && (labelmap[label] = str)
        label += 1
        str = ""
        continue
      end
      str = "$(str)$(i)"
    end
  end
  labelmap
end

end