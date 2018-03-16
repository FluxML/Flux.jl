module CIFAR10

using Colors

const NROWS = 32
const NCOLS = 32
const IMAGESIZE = NROWS * NCOLS
const LABELSIZE = 1
const N = 10000 # number of images per batch file (test included)

const dir = joinpath(@__DIR__, "../../deps/cifar10")
const RGB = Colors.RGB{Colors.N0f8}

file_batches = ["data_batch_1.bin",
		"data_batch_2.bin",
		"data_batch_3.bin",
		"data_batch_4.bin",
		"data_batch_5.bin"]
test_batch = "test_batch.bin"
meta_batch = "batches.meta.txt"

function load()
  mkpath(dir)
  cd(dir)
  
  for file in [file_batches..., test_batch, meta_batch]
    if !isfile(file)
      download("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", "cifar-10-binary.tar.gz")
      run(`tar -zxvf cifar-10-binary.tar.gz`)
      
      sub_dir = "cifar-10-batches-bin/"

      # move files and delete unnecessary folder
      for file in [file_batches..., test_batch, meta_batch]
        mv(string(sub_dir, file), file, remove_destination = true)
      end
      rm(sub_dir, recursive = true, force = true)

      return

    end
  end
end


function get_image(io::IO)
  rgb = [[], [], []]

  for i in 1:3
    rgb[i] = reinterpret(Colors.N0f8, read(io, UInt8, IMAGESIZE))
  end

  reshape([RGB(rgb[1][i], rgb[2][i], rgb[3][i]) for i=1:IMAGESIZE], NROWS, NCOLS)'
end

function get_label(io::IO)
  Int(read(io, LABELSIZE)[LABELSIZE])
end

get_io_buffers(files) = [IOBuffer(read(files[i])) for i=1:length(files)]

"""
    images()
    images(:test)

Load the CIFAR10 images.

Each image is a 32×2832 array of `RGB` colour values (see Colors.jl).

Returns the 50,000 training images by default; pass `:test` to retrieve the
10,000 test images.
"""
function images(set = :train)
  load()
  ios = get_io_buffers(set == :train ? file_batches : test_batch)

  # skip a label and get the image
  skip_and_get(io::IO) = (skip(io, LABELSIZE); get_image(io))

  [skip_and_get(io) for _ = 1:N for io in ios]
end

"""
    labels()
    labels(:test)

Load the labels corresponding to each of the images returned from `images()`.
Each label is a number from 0-9.

Returns the 50,000 training labels by default; pass `:test` to retrieve the
10,000 test labels.
"""
function labels(set = :train)
  load()
  ios = get_io_buffers(set == :train ? file_batches : test_batch)

  # get the label and skip an image
  get_and_skip(io::IO) = (l = get_label(io); skip(io, IMAGESIZE); l)

  [get_and_skip(io) for _ = 1:N for io in ios]
end

"""
    meta()

Returns the names of the classes corresponding to each of the labels returned
from `labels()`.
"""
function meta()
  readlines(meta_batch)[1:end-1]
end

end
